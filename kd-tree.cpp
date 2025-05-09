#include <vector>
#include <fstream>
#include <string_view>
#include <charconv>
#include <future>
#include <algorithm>
#include <functional>
#include <queue>
#include <cmath>
#include <sstream>
#include <iostream>
#include <chrono>

#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::string name; 
public:
    Timer(const std::string& _name) : name(_name) {
        start = std::chrono::high_resolution_clock::now();
    }

    void End() {
        end = std::chrono::high_resolution_clock::now();
    }

    void Report(){
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << name << ": " << duration << " µs\n";
    }
};

struct Point {
    double x, y;
};

inline double squared_distance(const Point a, const Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

std::vector<Point> load_points_from_file(const char *file_name) {
    std::ifstream file(file_name, std::ios::binary);
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(file_size);
    file.read(buffer.data(), file_size);
    
    std::string_view text(buffer.data(), buffer.size());
    const char* p = text.data();
    const char* end = p + text.size();
    
    unsigned int N = 0;
    auto [ptr, ec] = std::from_chars(p, end, N);
    p = ptr;
    
    std::vector<Point> points;
    points.reserve(N);
    while (p < end && std::isspace(*p))
        ++p;
    
    for (unsigned int i = 0; i < N; ++i) {
        double x = 0, y = 0;
        auto [ptr_x, ec_x] = std::from_chars(p, end, x);
        p = ptr_x;
        while (p < end && (std::isspace(*p) || *p == ','))
            ++p;
        auto [ptr_y, ec_y] = std::from_chars(p, end, y);
        p = ptr_y;
        while (p < end && (std::isspace(*p) || *p == '\n'))
            ++p;
        points.push_back({x, y});
    }
    
    return points;
}

struct KDNode {
    double splitVal;           // splitting value (coordinate of pivot)
    KDNode* left;
    KDNode* right;
    unsigned int pointIndex;   // index of pivot point in pts
    bool axis;                 // splitting axis: false for x, true for y
    KDNode(bool ax, double sv, unsigned int idx)
        : splitVal(sv), left(nullptr), right(nullptr), pointIndex(idx), axis(ax) {}
};

struct Neighbor {
    double dist;
    unsigned int index;
};

void knn_search_opt(KDNode* node, const Point& q, const std::vector<Point>& pts, const unsigned int k,
                    std::vector<Neighbor>& heap,
                    const std::function<bool(const Neighbor&, const Neighbor&)>& cmp) {
    if (!node) return;
    
    const Point& pivot = pts[node->pointIndex];
    double d_sq = squared_distance(q, pivot);
    
    if (heap.size() < k) {
        heap.push_back({d_sq, node->pointIndex});
        std::push_heap(heap.begin(), heap.end(), cmp);
    } else if (d_sq < heap.front().dist) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.back() = {d_sq, node->pointIndex};
        std::push_heap(heap.begin(), heap.end(), cmp);
    }
    
    bool axis = node->axis;
    double diff = (!axis) ? (q.x - node->splitVal) : (q.y - node->splitVal);
    
    KDNode* near = (diff < 0) ? node->left : node->right;
    KDNode* far  = (diff < 0) ? node->right : node->left;
    
    knn_search_opt(near, q, pts, k, heap, cmp);
    
    if (heap.size() < k || diff * diff < heap.front().dist)
        knn_search_opt(far, q, pts, k, heap, cmp);
}

template <typename Slice>
KDNode* build_kd_tree(const Slice &indices, const std::vector<Point> &pts, const unsigned int depth) {
    if (indices.size() == 0) return nullptr;
    
    // Choose splitting axis: false for x (when depth is even), true for y (when depth is odd)
    bool axis = (depth % 2) != 0;
    unsigned int median = indices.size() / 2;
    
    std::nth_element(indices.begin(), indices.begin() + median, indices.end(),
        [axis, &pts](unsigned int a, unsigned int b) {
            return (!axis) ? (pts[a].x < pts[b].x) : (pts[a].y < pts[b].y);
        }
    );
    
    unsigned int pivotIndex = indices[median];
    double splitVal = axis ? pts[pivotIndex].y : pts[pivotIndex].x;
    KDNode* node = new KDNode(axis, splitVal, pivotIndex);
    
    auto left_indices = indices.cut(0, median);
    auto right_indices = indices.cut(median + 1, indices.size());
    
    parlay::par_do(
        [&]() { node->left = build_kd_tree(left_indices, pts, depth + 1); },
        [&]() { node->right = build_kd_tree(right_indices, pts, depth + 1); }
    );
    
    return node;
}

std::vector<std::vector<Neighbor>> knn_search_all(KDNode* root, const std::vector<Point>& pts, const std::vector<Point>& queries, const unsigned int k) {
    std::vector<std::vector<Neighbor>> all_results(queries.size());
    parlay::parallel_for(0, queries.size(), [&](int i) {
        all_results[i].reserve(k);
    });

    auto cmp = [](const Neighbor& a, const Neighbor& b) {
        return a.dist < b.dist;
    };

    parlay::parallel_for(0, queries.size(), [&](int i) {
        knn_search_opt(root, queries[i], pts, k, all_results[i], cmp);
        std::sort(all_results[i].begin(), all_results[i].end(), [](const Neighbor& a, const Neighbor& b) {
            return a.dist < b.dist;
        });
    });

    return all_results;
}

void buffered_print_output(const unsigned int Q, std::vector<Point> &query_points, std::vector<std::vector<Neighbor>> &results) {
    constexpr size_t STDOUT_BUF_SIZE = 1 << 20;  // 1 MB.
    setvbuf(stdout, nullptr, _IOFBF, STDOUT_BUF_SIZE);

    constexpr size_t LOCAL_BUF_SIZE = 1 << 20;  // 1 MB.
    char* buf = new char[LOCAL_BUF_SIZE];
    char* out = buf;
    char* const buf_end = buf + LOCAL_BUF_SIZE;

    auto flush_buffer = [&]() {
        size_t len = out - buf;
        if (len > 0) {
            fwrite(buf, 1, len, stdout);
            out = buf;
        }
    };

    for (unsigned int q = 0; q < Q; ++q) {
        // Append "Query "
        {
            const char* s = "Query ";
            size_t len = std::strlen(s);
            if (out + len >= buf_end) flush_buffer();
            std::memcpy(out, s, len);
            out += len;
        }
        // Append q (the query index) using std::to_chars.
        {
            char numbuf[32];
            auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), q);
            if (res.ec != std::errc()) {
                const char* err = "ERR";
                size_t err_len = std::strlen(err);
                if (out + err_len >= buf_end) flush_buffer();
                std::memcpy(out, err, err_len);
                out += err_len;
            } else {
                size_t n = res.ptr - numbuf;
                if (out + n >= buf_end) flush_buffer();
                std::memcpy(out, numbuf, n);
                out += n;
            }
        }
        // Append " : ("
        {
            const char* s = " : (";
            size_t len = std::strlen(s);
            if (out + len >= buf_end) flush_buffer();
            std::memcpy(out, s, len);
            out += len;
        }
        // Append query_points[q].x
        {
            char numbuf[64];
            auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), query_points[q].x);
            if (res.ec != std::errc()) {
                const char* err = "ERR";
                size_t err_len = std::strlen(err);
                if (out + err_len >= buf_end) flush_buffer();
                std::memcpy(out, err, err_len);
                out += err_len;
            } else {
                size_t n = res.ptr - numbuf;
                if (out + n >= buf_end) flush_buffer();
                std::memcpy(out, numbuf, n);
                out += n;
            }
        }
        // Append ", "
        {
            const char* s = ", ";
            size_t len = std::strlen(s);
            if (out + len >= buf_end) flush_buffer();
            std::memcpy(out, s, len);
            out += len;
        }
        // Append query_points[q].y
        {
            char numbuf[64];
            auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), query_points[q].y);
            if (res.ec != std::errc()) {
                const char* err = "ERR";
                size_t err_len = std::strlen(err);
                if (out + err_len >= buf_end) flush_buffer();
                std::memcpy(out, err, err_len);
                out += err_len;
            } else {
                size_t n = res.ptr - numbuf;
                if (out + n >= buf_end) flush_buffer();
                std::memcpy(out, numbuf, n);
                out += n;
            }
        }
        // Append ")\n  kNN: "
        {
            const char* s = ")\n  kNN: ";
            size_t len = std::strlen(s);
            if (out + len >= buf_end) flush_buffer();
            std::memcpy(out, s, len);
            out += len;
        }
        // For each neighbor in results[q]
        for (const auto& neighbor : results[q]) {
            // Append "(dist2="
            {
                const char* s = "(dist2=";
                size_t len = std::strlen(s);
                if (out + len >= buf_end) flush_buffer();
                std::memcpy(out, s, len);
                out += len;
            }
            // Append neighbor.dist
            {
                char numbuf[64];
                auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), neighbor.dist);
                if (res.ec != std::errc()) {
                    const char* err = "ERR";
                    size_t err_len = std::strlen(err);
                    if (out + err_len >= buf_end) flush_buffer();
                    std::memcpy(out, err, err_len);
                    out += err_len;
                } else {
                    size_t n = res.ptr - numbuf;
                    if (out + n >= buf_end) flush_buffer();
                    std::memcpy(out, numbuf, n);
                    out += n;
                }
            }
            // Append ", idx="
            {
                const char* s = ", idx=";
                size_t len = std::strlen(s);
                if (out + len >= buf_end) flush_buffer();
                std::memcpy(out, s, len);
                out += len;
            }
            // Append neighbor.index
            {
                char numbuf[32];
                auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), neighbor.index);
                if (res.ec != std::errc()) {
                    const char* err = "ERR";
                    size_t err_len = std::strlen(err);
                    if (out + err_len >= buf_end) flush_buffer();
                    std::memcpy(out, err, err_len);
                    out += err_len;
                } else {
                    size_t n = res.ptr - numbuf;
                    if (out + n >= buf_end) flush_buffer();
                    std::memcpy(out, numbuf, n);
                    out += n;
                }
            }
            // Append ") "
            {
                const char* s = ") ";
                size_t len = std::strlen(s);
                if (out + len >= buf_end) flush_buffer();
                std::memcpy(out, s, len);
                out += len;
            }
        }
        // Append newline.
        {
            const char* s = "\n";
            size_t len = std::strlen(s);
            if (out + len >= buf_end) flush_buffer();
            std::memcpy(out, s, len);
            out += len;
        }
    }

    // Flush any remaining text in the buffer.
    if (out != buf) {
        fwrite(buf, 1, out - buf, stdout);
    }
    delete[] buf;
}

int main(int argc, char** argv) {
    //Timer total("Total");
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <query_file> <k>\n";
        return 1;
    }

    // Asynchronously load the data and query points.
    //Timer load_data("Load data");
    std::vector<Point> data_points, query_points;
    parlay::par_do(
          [&]() {data_points = load_points_from_file(argv[1]);},
          [&]() {query_points = load_points_from_file(argv[2]);}
    );
    const unsigned int k = std::stoi(argv[3]);
    const unsigned int N = data_points.size();
    const unsigned int Q = query_points.size();
    //load_data.End();

    //Timer iindices("Indices");
    parlay::sequence<unsigned int> indices = parlay::to_sequence(parlay::iota<unsigned int>(N));
    //iindices.End();

    
    // Build the KD-tree.
    //Timer build("Build");
    KDNode* root = build_kd_tree(indices.cut(0, N), data_points, 0);
    //build.End();
    
    // Perform kNN search for all queries in parallel.
    //Timer query("Query");
    auto results = knn_search_all(root, data_points, query_points, k);
    //query.End();
        
    //Timer print("Print");
    /*std::ostringstream oss;
    for (unsigned int q = 0; q < Q; q++) {
        oss << "Query " << q << " : (" 
            << query_points[q].x << ", " << query_points[q].y << ")\n  kNN: ";
        for (const auto& neighbor : results[q]) {
            oss << "(dist2=" << neighbor.dist << ", idx=" << neighbor.index << ") ";
        }
        oss << "\n";
    }
    std::cout << oss.str();*/
    buffered_print_output(Q, query_points, results);
    //print.End();
    
    //Timer dealoc("Deallocate");
    std::vector<KDNode*> stack;
    if (root) stack.push_back(root);
    while (!stack.empty()) {
        KDNode* node = stack.back();
        stack.pop_back();
        if (node->left)  stack.push_back(node->left);
        if (node->right) stack.push_back(node->right);
        delete node;
    }
    /*dealoc.End();    
    total.End();

    load_data.Report();
    iindices.Report();
    build.Report();
    query.Report();
    print.Report();
    dealoc.Report();
    total.Report();*/

    return 0;
}