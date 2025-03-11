// *********************************************************************************
// SmallList.h
// *********************************************************************************
#pragma once
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <new>

// ---------------------------------------------------------------------------------
// SmallList
// ---------------------------------------------------------------------------------
/// Stores a random-access sequence of elements similar to vector, but avoids 
/// heap allocations for small lists by using an SBO. T must be trivially
/// constructible and destructible (for people using C++11 and newer, you
/// can add a static assertion using type traits to ensure this).
template <class T>
class SmallList
{
public:
    // Creates an empty list.
    SmallList();

    // Creates a copy of the specified list.
    SmallList(const SmallList& other);

    // Copies the specified list.
    SmallList& operator=(const SmallList& other);

    // Destroys the list.
    ~SmallList();

    // Returns size() == 0.
    bool empty() const;

    // Returns the number of elements in the list.
    int size() const;

    // Returns the nth element.
    T& operator[](int n);

    // Returns the nth element in the list.
    const T& operator[](int n) const;

    // Clears the list.
    void clear();

    // Reserves space for n elements.
    void reserve(int n);

    // Resizes the list to contain 'n' elements.
    void resize(int num, const T& fill = T());

    // Inserts an element to the back of the list.
    void push_back(const T& element);

    /// Pops an element off the back of the list.
    T pop_back();

    // Swaps the contents of this list with the other.
    void swap(SmallList& other);

private:
    enum {fixed_cap = 64};
    struct ListData
    {
        ListData();
        T buf[fixed_cap];
        T* data;
        int num;
        int cap;
    };
    ListData ld;
};

// ---------------------------------------------------------------------------------
// FreeList
// ---------------------------------------------------------------------------------
/// Provides an indexed free list with constant-time removals from anywhere
/// in the list without invalidating indices. T must be trivially constructible 
/// and destructible.
template <class T>
class FreeList
{
public:
    /// Creates a new free list.
    FreeList();

    /// Inserts an element to the free list and returns an index to it.
    int insert(const T& element);

    // Removes the nth element from the free list.
    void erase(int n);

    // Removes all elements from the free list.
    void clear();

    // Returns the range of valid indices.
    int range() const;

    // Returns the nth element.
    T& operator[](int n);

    // Returns the nth element.
    const T& operator[](int n) const;

private:
    union FreeElement
    {
        T element;
        int next;
    };
    SmallList<FreeElement> data;
    int first_free;
};