#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>

#include "QuadTree.h"
#include "Print.h"
#include "UGrid.h"


int simulate_qtree(const int N, const int MAX_DEPTH, const int steps, const float dt, const float k, const bool save, const int SAVE_FREQUENCY, const std::string BASE_DIR, const float MAX_RADIUS, const float SPACING, const float polydispersity);
int simulate_grid(const int N, const int MAX_DEPTH, const int steps, const float dt, const float k, const bool save, const int SAVE_FREQUENCY, const std::string BASE_DIR, const float MAX_RADIUS, const float SPACING, const float polydispersity);


int main(int argc, char const *argv[])
{
    const int MAX_DEPTH = 32;
    const int steps = 500;
    const float dt = 0.01f;
    const float k = 500.0f;
    const bool save = false;
    const int SAVE_FREQUENCY = 80;
    const std::string BASE_DIR = "frames/";
    const float MAX_RADIUS  = 1.0f;
    const float SPACING = 3.5f * MAX_RADIUS;
    int performance1, performance2;  
    float polydispersity = 1.0;  // from 1.0 to infinity (how many times the biggest sphere can be compare with 1.0)

    std::cout << "poly" << "," << "N" << "," << "qtree" << "," << "grid" << "\n";
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 12; ++i) {
            int N = std::pow(2, i);
            performance1 = simulate_qtree(N, MAX_DEPTH, steps, dt, k, save, SAVE_FREQUENCY, BASE_DIR, MAX_RADIUS, SPACING, polydispersity);
            performance2 = simulate_grid(N, MAX_DEPTH, steps, dt, k, save, SAVE_FREQUENCY, BASE_DIR, MAX_RADIUS, SPACING, polydispersity);
            std::cout << polydispersity << "," << N << "," << performance1 << "," << performance2 << std::endl; 
        }
         polydispersity += 0.5;
    }

    return 0;
}


int simulate_grid(const int N, const int MAX_DEPTH, const int steps, const float dt, const float k, const bool save, const int SAVE_FREQUENCY, const std::string BASE_DIR, const float MAX_RADIUS, const float SPACING, const float polydispersity) {   
    std::vector<sphere> spheres(N);
    std::vector<float> ax(N, 0.0f);
    std::vector<float> ay(N, 0.0f);
    
    const float HEIGHT = std::sqrt(N) * SPACING * 4 * polydispersity;
    const float LENGTH = std::sqrt(N) * SPACING * 4 * polydispersity;
    const int COLS = static_cast<int>(std::ceil(std::sqrt(N)));
    const bool print = false;

    const int small = 2;
    const float big = 4.0*polydispersity;
    
    LGrid* grid = lgrid_create(big*MAX_RADIUS, big*MAX_RADIUS, small*MAX_RADIUS, small*MAX_RADIUS, 0.0f, 0.0f, LENGTH, HEIGHT);

    std::random_device rd;            // seed source
    std::mt19937 gen(rd());           // Mersenne Twister engine
    std::uniform_real_distribution<float> dis(0.0f, 2.0f);
    std::uniform_real_distribution<float> rad(MAX_RADIUS, polydispersity*MAX_RADIUS);

    // Initialize spheres in a grid.
    for (int i = 0; i < N; i++) {
        int row = i / COLS;
        int col = i % COLS;

        spheres[i].x = col * 2*polydispersity*MAX_RADIUS + polydispersity*MAX_RADIUS;
        spheres[i].y = row * 2*polydispersity*MAX_RADIUS + polydispersity*MAX_RADIUS;
        spheres[i].vx = -dis(gen);
        spheres[i].vy = -dis(gen);
        spheres[i].r = rad(gen);
        
        lgrid_insert(grid, i, spheres[i].x, spheres[i].y, spheres[i].r, spheres[i].r);
    }
    lgrid_optimize(grid);
    
    int frame = 0;
    if (save)
        save_frame_grid(BASE_DIR, spheres, grid, LENGTH, HEIGHT, frame);
    
    // Simulation loop.
    auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < steps; ++step) {
        
        // Compute collision forces.
        //#pragma omp parallel for private(qtree)
        for (int i = 0; i < N; i++) {
            SmallList<int> neighbors = lgrid_query(grid, spheres[i].x, spheres[i].y, spheres[i].r, spheres[i].r, i);
            
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
            float old_x = spheres[i].x;
            float old_y = spheres[i].y;

            spheres[i].vx += dt * ax[i];
            spheres[i].vy += dt * ay[i];
            spheres[i].x  += dt * spheres[i].vx;
            spheres[i].y  += dt * spheres[i].vy;
            
            // Reset forces.
            ax[i] = 0.0f; 
            ay[i] = 0.0f;
            
            lgrid_move(grid, i, old_x, old_y, spheres[i].x, spheres[i].y);
        }
        if (step % 2 == 0)
            lgrid_optimize(grid);

        if (save && step % SAVE_FREQUENCY == 0) {
            save_frame_grid(BASE_DIR, spheres, grid, LENGTH, HEIGHT, frame);
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    float elapsed_time = std::chrono::duration<float>(end - start).count();

    if (print) {
        std::cout << "N: " << N << ", Elapsed time: " << elapsed_time << " seconds" << std::endl;
        std::cout << "FPS: " << steps / elapsed_time << ", Performance: " << (N * steps) / elapsed_time << std::endl;
    }
    lgrid_destroy(grid);

    return (N * steps) / elapsed_time;
}


int simulate_qtree(const int N, const int MAX_DEPTH, const int steps, const float dt, const float k, const bool save, const int SAVE_FREQUENCY, const std::string BASE_DIR, const float MAX_RADIUS, const float SPACING, const float polydispersity) {   
    std::vector<sphere> spheres(N);
    std::vector<int> tree_id(N);
    std::vector<float> ax(N, 0.0f);
    std::vector<float> ay(N, 0.0f);
    
    const float HEIGHT = std::sqrt(N) * SPACING * 4 * polydispersity;
    const float LENGTH = std::sqrt(N) * SPACING * 4 * polydispersity;
    const int COLS = static_cast<int>(std::ceil(std::sqrt(N)));
    const bool print = false;
    
    Quadtree qtree(LENGTH, HEIGHT, MAX_DEPTH);

    std::random_device rd;            // seed source
    std::mt19937 gen(rd());           // Mersenne Twister engine
    std::uniform_real_distribution<float> dis(0.0f, 2.0f);
    std::uniform_real_distribution<float> rad(MAX_RADIUS, polydispersity*MAX_RADIUS);
    
    // Initialize spheres in a grid.
    for (int i = 0; i < N; i++) {
        int row = i / COLS;
        int col = i % COLS;

        spheres[i].x = col * 2*polydispersity*MAX_RADIUS + polydispersity*MAX_RADIUS;
        spheres[i].y = row * 2*polydispersity*MAX_RADIUS + polydispersity*MAX_RADIUS;
        spheres[i].vx = -dis(gen);
        spheres[i].vy = -dis(gen);
        spheres[i].r = rad(gen);
        
        float x1 = spheres[i].x - spheres[i].r; 
        float y1 = spheres[i].y - spheres[i].r;
        float x2 = spheres[i].x + spheres[i].r; 
        float y2 = spheres[i].y + spheres[i].r;
        
        int element = qtree.insert(i, x1, y1, x2, y2);
        tree_id[i] = element;
    }
    
    int frame = 0;
    if (save)
        save_frame_qtree(BASE_DIR, spheres, qtree, LENGTH, HEIGHT, frame);
    
    // Simulation loop.
    auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < steps; ++step) {
        
        // Compute collision forces.
        //#pragma omp parallel for private(qtree)
        for (int i = 0; i < N; i++) {
            SmallList<int> neighbors = qtree.query(
                spheres[i].x - 2 * spheres[i].r, 
                spheres[i].y - 2 * spheres[i].r,
                spheres[i].x + 2 * spheres[i].r, 
                spheres[i].y + 2 * spheres[i].r, 
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
        if (step % 2 == 0)
            qtree.cleanup();

        if (save && step % SAVE_FREQUENCY == 0) {
            save_frame_qtree(BASE_DIR, spheres, qtree, LENGTH, HEIGHT, frame);
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    float elapsed_time = std::chrono::duration<float>(end - start).count();

    if (print) {
        std::cout << "N: " << N << ", Elapsed time: " << elapsed_time << " seconds" << std::endl;
        std::cout << "FPS: " << steps / elapsed_time << ", Performance: " << (N * steps) / elapsed_time << std::endl;
    }

    return (N * steps) / elapsed_time;
}