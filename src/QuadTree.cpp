#include "SmallList.h"
#include "QuadTree.h"
#include <utility>
#include <cassert>

enum { max_elements = 8 };

// Forward declaration.
static void node_insert(Quadtree& tree, const QuadNodeData& node_data, int element);

// Updated intersect function to use floats.
static bool intersect(const float ltrb1[4], const float ltrb2[4])
{
  return ltrb2[0] <= ltrb1[2] && ltrb2[2] >= ltrb1[0] &&
         ltrb2[1] <= ltrb1[3] && ltrb2[3] >= ltrb1[1];
}

void leaf_insert(Quadtree& tree, const QuadNodeData& node_data, int element)
{
    QuadNode* node = &tree.nodes[node_data.index];

    // Insert the element node to the leaf.
    const QuadEltNode new_elt_node = { node->first_child, element };
    node->first_child = tree.elt_nodes.insert(new_elt_node);

    // If the leaf is full, split it.
    if (node->count == max_elements && node_data.depth < tree.max_depth)
    {
        // Pop off all the previous elements.
        SmallList<int> elts;
        while (node->first_child != -1)
        {
            const int index = node->first_child;
            node->first_child = tree.elt_nodes[node->first_child].next;
            elts.push_back(tree.elt_nodes[index].element);
            tree.elt_nodes.erase(index);
        }

        // Start by allocating 4 child nodes.
        if (node->first_child != -1)
            tree.free_node = tree.nodes[tree.free_node].first_child;
        else
        {
            node->first_child = static_cast<int>(tree.nodes.size());
            tree.nodes.resize(tree.nodes.size() + 4);
        }
        node = &tree.nodes[node_data.index];

        // Initialize the new child nodes.
        for (int j = 0; j < 4; ++j)
        {
            tree.nodes[node->first_child + j].first_child = -1;
            tree.nodes[node->first_child + j].count = 0;
        }

        // Transfer the elements in the former leaf node to its new children.
        node->count = -1;
        for (int j = 0; j < elts.size(); ++j)
            node_insert(tree, node_data, elts[j]);
    }
    else
        ++node->count;
}

static QuadNodeData child_data(float mx, float my, float sx, float sy, int index, int depth)
{
    const QuadNodeData cd = { { mx, my, sx, sy }, index, depth };
    return cd;
}

static QuadNodeList find_leaves(const Quadtree& tree, const QuadNodeData& root, const float rect[4])
{
    QuadNodeList leaves, to_process;
    to_process.push_back(root);
    while (to_process.size() > 0)
    {
        const QuadNodeData nd = to_process.pop_back();

        // If this node is a leaf, insert it to the list.
        if (tree.nodes[nd.index].count != -1)
            leaves.push_back(nd);
        else
        {
            // Otherwise push the children that intersect the rectangle.
            float mx = nd.rect.mid_x, my = nd.rect.mid_y;
            float sx = nd.rect.size_x, sy = nd.rect.size_y;
            float hx = sx / 2.0f, hy = sy / 2.0f;
            int fc = tree.nodes[nd.index].first_child;
            int dp = nd.depth + 1;

            if (rect[1] <= my)
            {
                if (rect[0] <= mx)
                    to_process.push_back(child_data(mx - hx, my - hy, hx, hy, fc + 0, dp));
                if (rect[2] > mx)
                    to_process.push_back(child_data(mx + hx, my - hy, hx, hy, fc + 1, dp));
            }
            if (rect[3] > my)
            {
                if (rect[0] <= mx)
                    to_process.push_back(child_data(mx - hx, my + hy, hx, hy, fc + 2, dp));
                if (rect[2] > mx)
                    to_process.push_back(child_data(mx + hx, my + hy, hx, hy, fc + 3, dp));
            }
        }
    }
    return leaves;
}

static void node_insert(Quadtree& tree, const QuadNodeData& node_data, int element)
{
    // Find the leaves and insert the element to all the leaves found.
    const QuadNodeList leaves = find_leaves(tree, node_data, tree.elts[element].ltrb);
    for (int j = 0; j < leaves.size(); ++j)
        leaf_insert(tree, leaves[j], element);
}

Quadtree::Quadtree(int width, int height, int imax_depth): 
    free_node(-1), max_depth(imax_depth)
{
    const QuadNode root_node = { -1, 0 };
    nodes.push_back(root_node);

    // Use float halves instead of bit-shifts.
    root_rect.size_x = width / 2.0f;
    root_rect.size_y = height / 2.0f;
    root_rect.mid_x = root_rect.size_x;
    root_rect.mid_y = root_rect.size_y;
}

int Quadtree::insert(int id, float x1, float y1, float x2, float y2)
{
    // Store the bounding box as floats.
    const QuadElt new_elt = { id, { x1, y1, x2, y2 } };
    const int element = elts.insert(new_elt);
    node_insert(*this, root_data(), element);
    return element;
}

void Quadtree::remove(int element)
{
    // Find the leaves.
    float rect[4] = { 0, 0, 0, 0 };
    // Use the float bounding box stored in the element.
    for (int i = 0; i < 4; ++i)
        rect[i] = elts[element].ltrb[i];

    const QuadNodeList leaves = find_leaves(*this, root_data(), rect);

    // For each leaf node, remove the element node.
    for (int j = 0; j < leaves.size(); ++j)
    {
        const QuadNodeData& nd = leaves[j];
        QuadNode& node = nodes[nd.index];

        // Walk the list until we find the element node.
        int* link = &node.first_child;
        while (*link != -1 && elt_nodes[*link].element != element)
        {
            link = &elt_nodes[*link].next;
            assert(*link != -1);
        }

        if (*link != -1)
        {
            // Remove the element node.
            const int elt_node_index = *link;
            *link = elt_nodes[*link].next;
            elt_nodes.erase(elt_node_index);
            --node.count;
        }
    }
    // Remove the element.
    elts.erase(element);
}

SmallList<int> Quadtree::query(float x1, float y1, float x2, float y2, int omit_element)
{
    float rect[4] = { x1, y1, x2, y2 };
    const QuadNodeList leaves = find_leaves(*this, root_data(), rect);

    SmallList<int> elements;
    temp.resize(elts.range(), false);
    for (int j = 0; j < leaves.size(); ++j)
    {
        const QuadNodeData& nd = leaves[j];
        QuadNode& node = nodes[nd.index];

        int elt_node_index = node.first_child;
        while (elt_node_index != -1)
        {
            const int element = elt_nodes[elt_node_index].element;
            if (!temp[element] && element != omit_element && intersect(elts[element].ltrb, rect))
            {
                elements.push_back(element);
                temp[element] = true;
            }
            elt_node_index = elt_nodes[elt_node_index].next;
        }
    }
    for (int j = 0; j < elements.size(); ++j)
        temp[elements[j]] = false;
    return elements;
}

void Quadtree::cleanup()
{
    SmallList<int> to_process;
    if (nodes[0].count == -1)
        to_process.push_back(0);

    while (to_process.size() > 0)
    {
        const int node_index = to_process.pop_back();
        QuadNode& node = nodes[node_index];

        int num_empty_leaves = 0;
        for (int j = 0; j < 4; ++j)
        {
            const int child_index = node.first_child + j;
            const QuadNode& child = nodes[child_index];
            if (child.count == 0)
                ++num_empty_leaves;
            else if (child.count == -1)
                to_process.push_back(child_index);
        }

        if (num_empty_leaves == 4)
        {
            nodes[node.first_child].first_child = free_node;
            free_node = node.first_child;

            node.first_child = -1;
            node.count = 0;
        }
    }
}

QuadNodeData Quadtree::root_data() const
{
    QuadNodeData rd = { 0 };
    rd.rect = root_rect;
    rd.index = 0;
    rd.depth = 0;
    return rd;
}