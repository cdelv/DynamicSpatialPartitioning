#pragma once
#include "SmallList.h"

// Represents a rectangle for the quadtree storing a center and half-size.
struct QuadCRect
{
    float mid_x;
    float mid_y;
    float size_x;
    float size_y;
};

// Represents an element in the quadtree.
struct QuadEltNode
{
    // Points to the next element in the leaf node. A value of -1 
    // indicates the end of the list.
    int next;

    // Stores the element index.
    int element;
};

// Represents an element in the quadtree.
struct QuadElt
{
    int id;
    // Changed from int to float
    float ltrb[4];
};

// Represents a node in the quadtree.
struct QuadNode
{
    // Points to the first child if this node is a branch or the first element
    // if this node is a leaf.
    int first_child;

    // Stores the number of elements in the node or -1 if it is not a leaf.
    int count;
};

// Stores temporary data about a node during a search.
struct QuadNodeData
{
    QuadCRect rect;
    int index;
    int depth;
};
typedef SmallList<QuadNodeData> QuadNodeList;

struct Quadtree
{
    // Creates a quadtree with the requested extents and max depth.
    Quadtree(int width, int height, int max_depth);

    // Inserts a new element to the tree.
    // Returns an index to the new element.
    int insert(int id, float x1, float y1, float x2, float y2);

    // Removes the specified element from the tree.
    void remove(int element);

    // Outputs a list of elements found in the specified rectangle.
    SmallList<int> query(float x1, float y1, float x2, float y2, int omit_element);

    // Return the data for the root node.
    QuadNodeData root_data() const;

    // Cleans up the tree, removing empty leaves.
    void cleanup();

    // Stores all the nodes in the quadtree. The first node in this
    // sequence is always the root.
    SmallList<QuadNode> nodes;

    // Temporary bitset used for queries.
    SmallList<char> temp;

    // Stores all the element data in the quadtree.
    FreeList<QuadElt> elts;

    // Stores all the elements in the quadtree.
    FreeList<QuadEltNode> elt_nodes;

    // Stores the width of the quadtree extents.
    QuadCRect root_rect;

    // Stores the first free node in the quadtree to be reclaimed as 4
    // contiguous nodes at once.
    int free_node;

    // Stores the maximum depth allowed for the quadtree.
    int max_depth;
};
