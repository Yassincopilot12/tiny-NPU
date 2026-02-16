// =============================================================================
// Processing Element for Systolic Array
// Passes data east (a) and south (b) with 1-cycle delay
// =============================================================================
`default_nettype none

module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        clear_acc,
    input  wire                        en,
    input  wire                        dtype_fp16,
    // INT8 data from west / north
    input  wire  signed [DATA_W-1:0]   a_in,
    input  wire  signed [DATA_W-1:0]   b_in,
    // INT8 data to east / south (delayed 1 cycle)
    output logic signed [DATA_W-1:0]   a_out,
    output logic signed [DATA_W-1:0]   b_out,
    // FP16 data from west / north
    input  wire  signed [15:0]         a_in_fp16,
    input  wire  signed [15:0]         b_in_fp16,
    // FP16 data to east / south (delayed 1 cycle)
    output logic signed [15:0]         a_out_fp16,
    output logic signed [15:0]         b_out_fp16,
    // Accumulated result (INT32 for both modes)
    output logic signed [ACC_W-1:0]    acc_out
);

    // INT8 pass-through registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
        end
    end

    // FP16 pass-through registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out_fp16 <= '0;
            b_out_fp16 <= '0;
        end else if (en) begin
            a_out_fp16 <= a_in_fp16;
            b_out_fp16 <= b_in_fp16;
        end
    end

    // INT8 MAC unit
    logic signed [ACC_W-1:0] int8_acc;
    mac_int8 #(
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_mac_int8 (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (clear_acc),
        .en        (en && !dtype_fp16),
        .a_in      (a_in),
        .b_in      (b_in),
        .acc_out   (int8_acc)
    );

    // FP16 MAC unit
    logic signed [ACC_W-1:0] fp16_acc;
    mac_fp16 #(
        .DATA_W (16),
        .ACC_W  (ACC_W)
    ) u_mac_fp16 (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (clear_acc),
        .en        (en && dtype_fp16),
        .a_in      (a_in_fp16),
        .b_in      (b_in_fp16),
        .acc_out   (fp16_acc)
    );

    // Mux output based on dtype
    assign acc_out = dtype_fp16 ? fp16_acc : int8_acc;

endmodule

`default_nettype wire
