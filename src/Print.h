#pragma once
#include "QuadTree.h"

#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

struct sphere {
    float x;
    float y;
    float vx;
    float vy;
    float r;    
};

void save_frame_qtree(const std::string& base_name, const std::vector<sphere>& spheres, const Quadtree& tree, const float LENGTH, const float HEIGHT, int &frame);
void save_to_csv(const std::string& base_name, const std::vector<sphere>& spheres, const int frame);
void save_simulation_box_vtk(const std::string& base_name, const float LENGTH, const float HEIGHT, const int frame);
void add_rectangle(const QuadCRect& rect, std::vector<std::array<float, 3>>& points, std::vector<std::vector<int>>& lines);
void traverse_quadtree(const Quadtree& tree, int node_index, const QuadCRect& rect, std::vector<std::array<float,3>>& points, std::vector<std::vector<int>>& lines);
void visualize_quadtree(const std::string& base_name, const Quadtree& tree, const int frame);