#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>

#include "QuadTree.h"
#include "Print.h"

int main(int argc, char const *argv[]) {   
    const int N = 500;
    const int MAX_DEPTH = 6;
    const int steps = 8000;
    const float dt = 0.01f;
    const float k = 500.0f;
    const bool save = true;  

    const int SAVE_FREQUENCY = 80;
    const std::string BASE_DIR = "frames/";
    
    std::vector<sphere> spheres(N);
    std::vector<int> tree_id(N);
    std::vector<float> ax(N, 0.0f);
    std::vector<float> ay(N, 0.0f);
    
    const float MAX_RADIUS  = 1.0f;
    const float SPACING = 2.0f * MAX_RADIUS;
    const float HEIGHT = N * SPACING / 8;
    const float LENGTH = N * SPACING / 8;
    const int COLS = static_cast<int>(std::ceil(std::sqrt(N)));
    
    Quadtree qtree(LENGTH, HEIGHT, MAX_DEPTH);

    std::random_device rd;            // seed source
    std::mt19937 gen(rd());           // Mersenne Twister engine
    std::uniform_real_distribution<float> dis(0.0f, 2.0f);
    
    // Initialize spheres in a grid.
    for (int i = 0; i < N; i++) {
        int row = i / COLS;
        int col = i % COLS;

        spheres[i].x = col * SPACING + MAX_RADIUS;
        spheres[i].y = row * SPACING + MAX_RADIUS;
        spheres[i].vx = -dis(gen);
        spheres[i].vy = -dis(gen);
        spheres[i].r = MAX_RADIUS;
        
        float x1 = spheres[i].x - spheres[i].r; 
        float y1 = spheres[i].y - spheres[i].r;
        float x2 = spheres[i].x + spheres[i].r; 
        float y2 = spheres[i].y + spheres[i].r;
        
        int element = qtree.insert(i, x1, y1, x2, y2);
        tree_id[i] = element;
    }
    
    int frame = 0;
    save_frame(BASE_DIR, spheres, qtree, LENGTH, HEIGHT, frame);
    
    // Simulation loop.
    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        
        // Compute collision forces.
        for (int i = 0; i < N; i++) {
            SmallList<int> neighbors = qtree.query(
                spheres[i].x - 2 * MAX_RADIUS, 
                spheres[i].y - 2 * MAX_RADIUS,
                spheres[i].x + 2 * MAX_RADIUS, 
                spheres[i].y + 2 * MAX_RADIUS, 
                i  // Omit the current sphere.
            );
            
            for (int j = 0; j < neighbors.size(); ++j) {
                const int id = neighbors[j];
                
                const float dx = spheres[id].x - spheres[i].x;
                const float dy = spheres[id].y - spheres[i].y;
                const float distance = std::hypot(dx, dy);
                
                const float minDistance = spheres[i].r + spheres[id].r;
                if (distance < minDistance && distance > 0.0f) {                    
                    const float nx = dx / distance;
                    const float ny = dy / distance;
                    const float overlap = minDistance - distance;
                    ax[i] -= k * overlap * nx;
                    ay[i] -= k * overlap * ny;
                }
            }

            // Force with walls.
            // Left wall: if (x - r < 0), then penetration = -(x - r).
            float s = spheres[i].x - spheres[i].r;
            if (s < 0) {
                ax[i] += k * (-s);
            }

            // Right wall: if (x + r > LENGTH), then penetration = (x + r) - LENGTH.
            s = spheres[i].x + spheres[i].r - LENGTH;
            if (s > 0) {
                ax[i] -= k * s;
            }

            // Bottom wall: if (y - r < 0), then penetration = -(y - r).
            s = spheres[i].y - spheres[i].r;
            if (s < 0) {
                ay[i] += k * (-s);
            }

            // Top wall: if (y + r > HEIGHT), then penetration = (y + r) - HEIGHT.
            s = spheres[i].y + spheres[i].r - HEIGHT;
            if (s > 0) {
                ay[i] -= k * s;
            }
        }
        
        // Update positions and velocities.
        for (int i = 0; i < N; i++) {
            spheres[i].vx += dt * ax[i];
            spheres[i].vy += dt * ay[i];
            spheres[i].x  += dt * spheres[i].vx;
            spheres[i].y  += dt * spheres[i].vy;
            
            // Reset forces.
            ax[i] = 0.0f; 
            ay[i] = 0.0f;
            
            // Update quadtree with new sphere positions.
            qtree.remove(tree_id[i]);
            const float x1 = spheres[i].x - spheres[i].r; 
            const float y1 = spheres[i].y - spheres[i].r;
            const float x2 = spheres[i].x + spheres[i].r; 
            const float y2 = spheres[i].y + spheres[i].r;
            const int element = qtree.insert(i, x1, y1, x2, y2);
            tree_id[i] = element;
        }
        qtree.cleanup();

        if (save && step % SAVE_FREQUENCY == 0) {
            save_frame(BASE_DIR, spheres, qtree, LENGTH, HEIGHT, frame);
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    float elapsed_time = std::chrono::duration<float>(end - start).count();
    std::cout << "N: " << N << ", Elapsed time: " << elapsed_time << " seconds" << std::endl;
    std::cout << "FPS: " << steps / elapsed_time << ", Performance: " << (N * steps) / elapsed_time << std::endl;

    return 0;
}