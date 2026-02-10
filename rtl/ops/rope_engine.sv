// =============================================================================
// rope_engine.sv - Rotary Position Embedding (RoPE) engine
// Applies rotary position embeddings to Q or K vectors in SRAM0.
// Reads sin/cos tables from SRAM1.
//
// Formula (per position p, dim pair i):
//   q_rot[2i]   = (q[2i]*cos[p,i] - q[2i+1]*sin[p,i]) >> 7
//   q_rot[2i+1] = (q[2i]*sin[p,i] + q[2i+1]*cos[p,i]) >> 7
//
// 4 cycles per dimension pair: RD_EVEN -> RD_ODD -> WR_EVEN -> WR_ODD
// Total cycles: 4 * (HEAD_DIM/2) * num_rows
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module rope_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Command interface
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        src_base,     // SRAM0: Q or K [num_rows, head_dim]
    input  logic [15:0]        dst_base,     // SRAM0: output (can be same as src for in-place)
    input  logic [15:0]        num_rows,     // S (sequence length)
    input  logic [15:0]        head_dim,     // HEAD_DIM (must be even)
    input  logic [15:0]        pos_offset,   // starting position (0 for prefill)
    input  logic [15:0]        sin_base,     // SRAM1: sin table [MAX_SEQ, HEAD_DIM/2]
    input  logic [15:0]        cos_base,     // SRAM1: cos table [MAX_SEQ, HEAD_DIM/2]

    // SRAM0 read port (Q/K elements)
    output logic               sram_rd0_en,
    output logic [15:0]        sram_rd0_addr,
    input  logic [DATA_W-1:0]  sram_rd0_data,

    // SRAM1 read port (sin/cos tables)
    output logic               sram_rd1_en,
    output logic [15:0]        sram_rd1_addr,
    input  logic [DATA_W-1:0]  sram_rd1_data,

    // SRAM0 write port (rotated output)
    output logic               sram_wr_en,
    output logic [15:0]        sram_wr_addr,
    output logic [DATA_W-1:0]  sram_wr_data,

    // Status
    output logic               busy,
    output logic               done
);

    // ----------------------------------------------------------------
    // FSM States
    // ----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_RD_EVEN,    // Read q[row, 2i] from SRAM0, cos[pos, i] from SRAM1
        S_RD_ODD,     // Read q[row, 2i+1] from SRAM0, sin[pos, i] from SRAM1
        S_WR_EVEN,    // Compute and write q_rot[2i]
        S_WR_ODD,     // Compute and write q_rot[2i+1], advance
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // ----------------------------------------------------------------
    // Registered command parameters
    // ----------------------------------------------------------------
    logic [15:0] r_src_base, r_dst_base;
    logic [15:0] r_num_rows, r_head_dim, r_pos_offset;
    logic [15:0] r_sin_base, r_cos_base;
    logic [15:0] r_half_dim;   // head_dim / 2

    // Row and pair counters
    logic [15:0] r_row;        // current row [0, num_rows)
    logic [15:0] r_pair;       // current pair [0, head_dim/2)

    // Latched values for rotation
    logic signed [7:0] r_even_val;   // q[row, 2i]
    logic signed [7:0] r_odd_val;    // q[row, 2i+1]
    logic signed [7:0] r_cos_val;    // cos[pos, i]
    logic signed [7:0] r_sin_val;    // sin[pos, i]

    // Computed rotated values
    logic signed [15:0] rot_even;    // (even*cos - odd*sin + 64) >> 7
    logic signed [15:0] rot_odd;     // (even*sin + odd*cos + 64) >> 7
    logic signed [7:0]  rot_even_clamp;
    logic signed [7:0]  rot_odd_clamp;

    // ----------------------------------------------------------------
    // FSM transition
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= state_nxt;

    logic last_pair;
    logic last_row;
    assign last_pair = (r_pair == r_half_dim - 16'd1);
    assign last_row  = (r_row == r_num_rows - 16'd1);

    always_comb begin
        state_nxt = state;
        case (state)
            S_IDLE:     if (cmd_valid) state_nxt = S_RD_EVEN;
            S_RD_EVEN:  state_nxt = S_RD_ODD;
            S_RD_ODD:   state_nxt = S_WR_EVEN;
            S_WR_EVEN:  state_nxt = S_WR_ODD;
            S_WR_ODD: begin
                if (last_pair && last_row)
                    state_nxt = S_DONE;
                else
                    state_nxt = S_RD_EVEN;
            end
            S_DONE:     state_nxt = S_IDLE;
            default:    state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Command capture
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_src_base   <= '0;
            r_dst_base   <= '0;
            r_num_rows   <= '0;
            r_head_dim   <= '0;
            r_pos_offset <= '0;
            r_sin_base   <= '0;
            r_cos_base   <= '0;
            r_half_dim   <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_src_base   <= src_base;
            r_dst_base   <= dst_base;
            r_num_rows   <= num_rows;
            r_head_dim   <= head_dim;
            r_pos_offset <= pos_offset;
            r_sin_base   <= sin_base;
            r_cos_base   <= cos_base;
            r_half_dim   <= head_dim >> 1;
        end
    end

    // ----------------------------------------------------------------
    // Row and pair counters
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_row  <= '0;
            r_pair <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_row  <= '0;
            r_pair <= '0;
        end else if (state == S_WR_ODD && state_nxt == S_RD_EVEN) begin
            if (last_pair) begin
                r_pair <= '0;
                r_row  <= r_row + 16'd1;
            end else begin
                r_pair <= r_pair + 16'd1;
            end
        end
    end

    // ----------------------------------------------------------------
    // SRAM address computation
    // ----------------------------------------------------------------
    logic [15:0] src_even_addr, src_odd_addr;
    logic [15:0] dst_even_addr, dst_odd_addr;
    logic [15:0] cos_addr, sin_addr;
    logic [15:0] pos_idx;     // actual position = row + pos_offset

    assign pos_idx       = r_row + r_pos_offset;
    assign src_even_addr = r_src_base + r_row * r_head_dim + (r_pair << 1);
    assign src_odd_addr  = r_src_base + r_row * r_head_dim + (r_pair << 1) + 16'd1;
    assign dst_even_addr = r_dst_base + r_row * r_head_dim + (r_pair << 1);
    assign dst_odd_addr  = r_dst_base + r_row * r_head_dim + (r_pair << 1) + 16'd1;
    assign cos_addr      = r_cos_base + pos_idx * r_half_dim + r_pair;
    assign sin_addr      = r_sin_base + pos_idx * r_half_dim + r_pair;

    // ----------------------------------------------------------------
    // SRAM read control
    // ----------------------------------------------------------------
    always_comb begin
        sram_rd0_en   = 1'b0;
        sram_rd0_addr = '0;
        sram_rd1_en   = 1'b0;
        sram_rd1_addr = '0;

        if (state == S_RD_EVEN) begin
            sram_rd0_en   = 1'b1;
            sram_rd0_addr = src_even_addr;
            sram_rd1_en   = 1'b1;
            sram_rd1_addr = cos_addr;
        end else if (state == S_RD_ODD) begin
            sram_rd0_en   = 1'b1;
            sram_rd0_addr = src_odd_addr;
            sram_rd1_en   = 1'b1;
            sram_rd1_addr = sin_addr;
        end
    end

    // ----------------------------------------------------------------
    // Latch read values
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_RD_ODD) begin
            // By now SRAM0/SRAM1 data from RD_EVEN is available
            r_even_val <= signed'(sram_rd0_data);
            r_cos_val  <= signed'(sram_rd1_data);
        end
        if (state == S_WR_EVEN) begin
            // By now SRAM0/SRAM1 data from RD_ODD is available
            r_odd_val <= signed'(sram_rd0_data);
            r_sin_val <= signed'(sram_rd1_data);
        end
    end

    // ----------------------------------------------------------------
    // Rotation computation
    // rot_even = (even*cos - odd*sin + 64) >> 7
    // rot_odd  = (even*sin + odd*cos + 64) >> 7
    // ----------------------------------------------------------------
    always_comb begin
        rot_even = (16'(signed'(r_even_val)) * 16'(signed'(r_cos_val))
                  - 16'(signed'(r_odd_val))  * 16'(signed'(r_sin_val))
                  + 16'sd64) >>> 7;
        rot_odd  = (16'(signed'(r_even_val)) * 16'(signed'(r_sin_val))
                  + 16'(signed'(r_odd_val))  * 16'(signed'(r_cos_val))
                  + 16'sd64) >>> 7;

        // Clamp to int8
        if (rot_even > 16'sd127)       rot_even_clamp = 8'sd127;
        else if (rot_even < -16'sd128) rot_even_clamp = -8'sd128;
        else                           rot_even_clamp = rot_even[7:0];

        if (rot_odd > 16'sd127)        rot_odd_clamp = 8'sd127;
        else if (rot_odd < -16'sd128)  rot_odd_clamp = -8'sd128;
        else                           rot_odd_clamp = rot_odd[7:0];
    end

    // ----------------------------------------------------------------
    // SRAM write control
    // ----------------------------------------------------------------
    assign sram_wr_en   = (state == S_WR_EVEN) || (state == S_WR_ODD);
    assign sram_wr_addr = (state == S_WR_EVEN) ? dst_even_addr : dst_odd_addr;
    assign sram_wr_data = (state == S_WR_EVEN) ? rot_even_clamp : rot_odd_clamp;

    // ----------------------------------------------------------------
    // Status and handshake
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
