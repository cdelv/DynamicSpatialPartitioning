#include "Print.h"

void save_frame(const std::string& base_name, const std::vector<sphere>& spheres, const Quadtree& tree, const float LENGTH, const float HEIGHT, int &frame) {
    save_simulation_box_vtk(base_name, LENGTH, HEIGHT, frame);
    save_to_csv(base_name, spheres, frame);
    visualize_quadtree(base_name, tree, frame);
    frame++;
}

void save_to_csv(const std::string& base_name, const std::vector<sphere>& spheres, const int frame) {    
    std::ostringstream filename;
    filename << base_name << "particles_" << std::setw(5) << std::setfill('0') << frame << ".csv";
    std::ofstream file(filename.str());
    
    // Write header
    file << "x,y,z,r" << std::endl;
    
    // Write data for each sphere
    for (const auto& s : spheres) {
        file << s.x << "," << s.y << "," << 0.0f << "," << s.r << std::endl;
    }
    
    file.close();
}

void save_simulation_box_vtk(const std::string& base_name, const float LENGTH, const float HEIGHT, const int frame) {
    std::ostringstream filename;
    filename << base_name << "box_" << std::setw(5) << std::setfill('0') << frame << ".vtk";
    std::ofstream file(filename.str());
    
    // Write a legacy VTK file header
    file << "# vtk DataFile Version 3.0\n";
    file << "Simulation Box\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";
    
    // Define 5 points: the 4 corners and repeat the first point to close the loop.
    file << "POINTS 5 float\n";
    file << "0 0 0\n";
    file << LENGTH << " 0 0\n";
    file << LENGTH << " " << HEIGHT << " 0\n";
    file << "0 " << HEIGHT << " 0\n";
    file << "0 0 0\n"; // Repeat the first point to close the loop.
    
    // Define a polyline cell that uses these 5 points.
    // "LINES 1 6" means one polyline with 6 integers: the first is the number of points (5), followed by the indices.
    file << "LINES 1 6\n";
    file << "5 0 1 2 3 4\n";
    
    file.close();
}

void add_rectangle(const QuadCRect& rect, std::vector<std::array<float, 3>>& points, std::vector<std::vector<int>>& lines) {
    // Compute boundaries from center and half-sizes.
    float xmin = rect.mid_x - rect.size_x;
    float xmax = rect.mid_x + rect.size_x;
    float ymin = rect.mid_y - rect.size_y;
    float ymax = rect.mid_y + rect.size_y;

    int start = points.size();
    points.push_back({xmin, ymin, 0.0f});
    points.push_back({xmax, ymin, 0.0f});
    points.push_back({xmax, ymax, 0.0f});
    points.push_back({xmin, ymax, 0.0f});
    // Repeat the first point to close the rectangle.
    points.push_back({xmin, ymin, 0.0f});

    // Store the indices as a polyline cell.
    lines.push_back({ start, start + 1, start + 2, start + 3, start + 4 });
}

void traverse_quadtree(const Quadtree& tree, int node_index, const QuadCRect& rect, std::vector<std::array<float,3>>& points, std::vector<std::vector<int>>& lines) {
    // Add this node's rectangle.
    add_rectangle(rect, points, lines);

    const QuadNode& node = tree.nodes[node_index];
    // If the node is internal (count == -1), then it has 4 children.
    if (node.count == -1)
    {
        // Child rectangles: split the parent's half-sizes.
        float hx = rect.size_x / 2.0f;
        float hy = rect.size_y / 2.0f;
        QuadCRect child_rect;

        // Child 0: bottom-left.
        child_rect.mid_x = rect.mid_x - hx;
        child_rect.mid_y = rect.mid_y - hy;
        child_rect.size_x = hx;
        child_rect.size_y = hy;
        traverse_quadtree(tree, node.first_child + 0, child_rect, points, lines);

        // Child 1: bottom-right.
        child_rect.mid_x = rect.mid_x + hx;
        child_rect.mid_y = rect.mid_y - hy;
        traverse_quadtree(tree, node.first_child + 1, child_rect, points, lines);

        // Child 2: top-left.
        child_rect.mid_x = rect.mid_x - hx;
        child_rect.mid_y = rect.mid_y + hy;
        traverse_quadtree(tree, node.first_child + 2, child_rect, points, lines);

        // Child 3: top-right.
        child_rect.mid_x = rect.mid_x + hx;
        child_rect.mid_y = rect.mid_y + hy;
        traverse_quadtree(tree, node.first_child + 3, child_rect, points, lines);
    }
}

void visualize_quadtree(const std::string& base_name, const Quadtree& tree, const int frame) {
    std::ostringstream filename;
    filename << base_name << "quadtree_" << std::setw(5) << std::setfill('0') << frame << ".vtk";
    std::ofstream file(filename.str());

    // Vectors to store points and polyline cells.
    std::vector<std::array<float,3>> points;
    std::vector<std::vector<int>> lines;

    // Start traversal at the root node (index 0) using tree.root_rect.
    traverse_quadtree(tree, 0, tree.root_rect, points, lines);

    // Write VTK legacy header.
    file << "# vtk DataFile Version 3.0\n";
    file << "Quadtree Visualization\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";

    // Write the points.
    file << "POINTS " << points.size() << " float\n";
    for (const auto& pt : points)
    {
        file << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }

    // Compute total number of integers to be written for the lines.
    // For each polyline, we write 1 integer for the number of points plus that many indices.
    int total_line_size = 0;
    for (const auto& line : lines)
    {
        total_line_size += (1 + line.size());
    }

    file << "LINES " << lines.size() << " " << total_line_size << "\n";
    for (const auto& line : lines)
    {
        file << line.size();
        for (int idx : line)
        {
            file << " " << idx;
        }
        file << "\n";
    }
    file.close();
}