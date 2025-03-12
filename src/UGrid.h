#pragma once
#include "SmallList.h"
#include <algorithm> 
#include <immintrin.h> 

//------------------------------------------------------------------------------
// Begin SIMD Intrinsics replacements
//------------------------------------------------------------------------------

// Create a __m128 vector with the four floats in order.
inline __m128 simd_create4f(float a, float b, float c, float d) {
    return _mm_setr_ps(a, b, c, d);
}

// Multiply two __m128 vectors elementwise.
inline __m128 simd_mul4f(__m128 a, __m128 b) {
    return _mm_mul_ps(a, b);
}

// Add and subtract operations.
inline __m128 simd_add4f(__m128 a, __m128 b) {
    return _mm_add_ps(a, b);
}

inline __m128 simd_sub4f(__m128 a, __m128 b) {
    return _mm_sub_ps(a, b);
}

// Load a __m128 from memory (unaligned load is used for safety).
inline __m128 simd_load4f(const float* p) {
    return _mm_loadu_ps(p);
}

inline __m128 simd_loadu4f(const float* p) {
    return _mm_loadu_ps(p);
}

// Create a __m128 with all four elements equal to the scalar.
inline __m128 simd_scalar4f(float s) {
    return _mm_set1_ps(s);
}

// Create a __m128i with all four elements equal to the integer.
inline __m128i simd_scalar4i(int s) {
    return _mm_set1_epi32(s);
}

// Convert a __m128 to __m128i using truncation.
inline __m128i simd_ftoi4f(__m128 a) {
    return _mm_cvttps_epi32(a);
}

// Return a __m128i vector of zeros.
inline __m128i simd_zero4i() {
    return _mm_setzero_si128();
}

// Create a __m128i vector from four integers.
inline __m128i simd_create4i(int a, int b, int c, int d) {
    return _mm_setr_epi32(a, b, c, d);
}

// Clamp an integer vector between min and max values.
inline __m128i simd_clamp4i(__m128i x, __m128i min_val, __m128i max_val) {
    x = _mm_max_epi32(x, min_val);
    x = _mm_min_epi32(x, max_val);
    return x;
}

// Define our own vector types so that we can access the results in an array.
struct SimdVec4f {
    float data[4];
};

struct SimdVec4i {
    int data[4];
};

// Store a __m128 into a SimdVec4f.
inline SimdVec4f simd_store4f(__m128 v) {
    SimdVec4f r;
    _mm_storeu_ps(r.data, v);
    return r;
}

// Store a __m128i into a SimdVec4i.
inline SimdVec4i simd_store4i(__m128i v) {
    SimdVec4i r;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(r.data), v);
    return r;
}

// A simple (scalar) rectangle-intersection test.
// The __m128 is assumed to contain [left, top, right, bottom].
inline bool simd_rect_intersect4f(__m128 a, __m128 b) {
    float a_arr[4], b_arr[4];
    _mm_storeu_ps(a_arr, a);
    _mm_storeu_ps(b_arr, b);
    return (a_arr[0] <= b_arr[2] && a_arr[2] >= b_arr[0] &&
            a_arr[1] <= b_arr[3] && a_arr[3] >= b_arr[1]);
}


typedef SimdVec4f  SimdVec4f;   ///< Now our own four-component SIMD vector type for floats
typedef SimdVec4f  SimdVecf;    ///< Typedef corresponding to a platform-independent SIMD vector type
typedef SimdVec4i  SimdVec4i;   ///< Our four-component SIMD integer type
typedef SimdVec4i  SimdVeci;

// New overload: allow loading from a SimdVec4f pointer.
inline __m128 simd_load4f(const SimdVec4f* p) {
    return _mm_loadu_ps(p->data);
}


struct LGridQuery4
{
    // Stores the resulting elements of the SIMD query.
    SmallList<int> elements[4];
};

struct LGridElt
{
    // Stores the index to the next element in the loose cell using an indexed SLL.
    int next;

    // Stores the ID of the element. This can be used to associate external
    // data to the element.
    int id;

    // Stores the center of the element.
    float mx, my;

    // Stores the half-size of the element relative to the upper-left corner
    // of the grid.
    float hx, hy;
};

struct LGridLooseCell
{
    // Stores the extents of the grid cell relative to the upper-left corner
    // of the grid which expands and shrinks with the elements inserted and 
    // removed.
    float rect[4];

    // Stores the index to the first element using an indexed SLL.
    int head;
};

struct LGridLoose
{
    // Stores all the cells in the loose grid.
    LGridLooseCell* cells;

    // Stores the number of columns, rows, and cells in the loose grid.
    int num_cols, num_rows, num_cells;

    // Stores the inverse size of a loose cell.
    float inv_cell_w, inv_cell_h;
};

struct LGridTightCell
{
    // Stores the index to the next loose cell in the grid cell.
    int next;

    // Stores the position of the loose cell in the grid.
    int lcell;
};

struct LGridTight
{
    // Stores all the tight cell nodes in the grid.
    FreeList<LGridTightCell> cells;

    // Stores the tight cell heads.
    int* heads;

    // Stores the number of columns, rows, and cells in the tight grid.
    int num_cols, num_rows, num_cells;

    // Stores the inverse size of a tight cell.
    float inv_cell_w, inv_cell_h;
};

struct LGrid
{
    // Stores the tight cell data for the grid.
    LGridTight tight;

    // Stores the loose cell data for the grid.
    LGridLoose loose;

    // Stores all the elements in the grid.
    FreeList<LGridElt> elts;

    // Stores the number of elements in the grid.
    int num_elts;

    // Stores the upper-left corner of the grid.
    float x, y;

    // Stores the size of the grid.
    float w, h;
};

// Creates a loose grid encompassing the specified extents using the specified cell 
// size. Elements inserted to the loose grid are only inserted in one cell, but the
// extents of each cell are allowed to expand and shrink. To avoid requiring every
// loose cell to be checked during a search, a second grid of tight cells referencing
// the loose cells is stored.
LGrid* lgrid_create(float lcell_w, float lcell_h, float tcell_w, float tcell_h,
                    float l, float t, float r, float b);

// Destroys the grid.
void lgrid_destroy(LGrid* grid);

// Returns the grid cell index for the specified position.
int lgrid_lcell_idx(LGrid* grid, float x, float y);

// Inserts an element to the grid.
void lgrid_insert(LGrid* grid, int id, float mx, float my, float hx, float hy);

// Removes an element from the grid.
void lgrid_remove(LGrid* grid, int id, float mx, float my);

// Moves an element in the grid from the former position to the new one.
void lgrid_move(LGrid* grid, int id, float prev_mx, float prev_my, float mx, float my);

// Returns all the element IDs that intersect the specified rectangle excluding elements
// with the specified ID to omit.
SmallList<int> lgrid_query(const LGrid* grid, float mx, float my, float hx, float hy, int omit_id);

// Returns all the element IDs that intersect the specified 4 rectangles excluding elements
// with the specified IDs to omit.
LGridQuery4 lgrid_query4(const LGrid* grid, const SimdVec4f* mx4, const SimdVec4f* my4, 
                         const SimdVec4f* hx4, const SimdVec4f* hy4, const SimdVec4i* omit_id4);

// Returns true if the specified rectangle is inside the grid boundaries.
bool lgrid_in_bounds(const LGrid* grid, float mx, float my, float hx, float hy);

// Optimizes the grid, shrinking bounding boxes in response to removed elements and
// rearranging the memory of the grid to allow cache-friendly cell traversal.
void lgrid_optimize(LGrid* grid);