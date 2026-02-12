// =============================================================================
// DDR Memory Map for Graph Mode
// All addresses are byte offsets from DDR base (0x0000_0000 in sim)
// Each region is 64-byte aligned
// =============================================================================
#ifndef DDR_GRAPH_H
#define DDR_GRAPH_H

#include <cstdint>

// DDR regions for graph mode artifacts
static constexpr uint32_t DDR_GRAPH_PROG_BASE    = 0x00100000; // Program instructions
static constexpr uint32_t DDR_GRAPH_TDESC_BASE   = 0x00200000; // Tensor descriptor table
static constexpr uint32_t DDR_GRAPH_DATA_BASE    = 0x00300000; // Weight/bias data
static constexpr uint32_t DDR_GRAPH_IO_BASE      = 0x00400000; // Input/output tensors
static constexpr uint32_t DDR_GRAPH_SCRATCH_BASE = 0x00500000; // Scratch space

// Maximum sizes
static constexpr uint32_t DDR_GRAPH_PROG_MAX     = 0x00010000; // 64KB program
static constexpr uint32_t DDR_GRAPH_TDESC_MAX    = 0x00002000; // 8KB descriptors (256 x 32B)
static constexpr uint32_t DDR_GRAPH_DATA_MAX     = 0x00100000; // 1MB data
static constexpr uint32_t DDR_GRAPH_IO_MAX       = 0x00010000; // 64KB IO

#endif // DDR_GRAPH_H
