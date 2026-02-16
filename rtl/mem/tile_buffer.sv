`default_nettype none

module tile_buffer #(
    parameter int BANK_DEPTH = 256,  // ARRAY_M * ARRAY_N for 16x16
    parameter int DATA_W     = 8
)(
    input  wire                           clk,
    input  wire                           rst_n,

    // DMA/prefetch write port (writes to prefetch_bank)
    input  wire                           dma_wr_en,
    input  wire  [$clog2(BANK_DEPTH)-1:0] dma_wr_addr,
    input  wire  [DATA_W-1:0]             dma_wr_data,

    // Compute read port (reads from compute_bank)
    input  wire                           compute_rd_en,
    input  wire  [$clog2(BANK_DEPTH)-1:0] compute_rd_addr,
    output logic [DATA_W-1:0]             compute_rd_data,

    // Bank swap control
    input  wire                           swap,           // pulse to swap banks

    // Handshake signals
    output logic                          tile_done,      // compute bank has been fully read
    output logic                          next_tile_ready // prefetch bank is loaded and ready
);

    localparam int ADDR_W = $clog2(BANK_DEPTH);

    // ---------------------------------------------------------------
    // Two banks of storage
    // ---------------------------------------------------------------
    logic [DATA_W-1:0] bank0 [0:BANK_DEPTH-1];
    logic [DATA_W-1:0] bank1 [0:BANK_DEPTH-1];

    // Active bank for compute reads (0 or 1)
    logic active_bank;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            active_bank <= 1'b0;
        else if (swap)
            active_bank <= ~active_bank;
    end

    // ---------------------------------------------------------------
    // DMA writes to prefetch bank (opposite of active)
    // ---------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (dma_wr_en) begin
            if (active_bank)
                bank0[dma_wr_addr] <= dma_wr_data;
            else
                bank1[dma_wr_addr] <= dma_wr_data;
        end
    end

    // ---------------------------------------------------------------
    // Compute reads from active bank (registered output, 1-cycle latency)
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            compute_rd_data <= '0;
        else if (compute_rd_en) begin
            if (active_bank)
                compute_rd_data <= bank1[compute_rd_addr];
            else
                compute_rd_data <= bank0[compute_rd_addr];
        end
    end

    // ---------------------------------------------------------------
    // Handshake placeholders
    // ---------------------------------------------------------------

    // next_tile_ready: registered flag, toggles on each swap pulse
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            next_tile_ready <= 1'b0;
        else if (swap)
            next_tile_ready <= ~next_tile_ready;
    end

    // tile_done: placeholder, tied low (GEMM controller manages externally)
    assign tile_done = 1'b0;

endmodule

`default_nettype wire
