// ---------------------------------------------------------------------------------
// SmallList Implementation
// ---------------------------------------------------------------------------------
#include "SmallList.h"

template <class T>
SmallList<T>::ListData::ListData(): data(buf), num(0), cap(fixed_cap) {}

template <class T>
SmallList<T>::SmallList() {}

template <class T>
SmallList<T>::SmallList(const SmallList& other)
{
    if (other.ld.cap == fixed_cap)
    {
        ld = other.ld;
        ld.data = ld.buf;
    }
    else
    {
        reserve(other.ld.num);
        for (int j=0; j < other.size(); ++j)
            ld.data[j] = other.ld.data[j];
        ld.num = other.ld.num;
        ld.cap = other.ld.cap;
    }
}

template <class T>
SmallList<T>& SmallList<T>::operator=(const SmallList<T>& other)
{
    SmallList(other).swap(*this);
    return *this;
}

template <class T>
SmallList<T>::~SmallList()
{
    if (ld.data != ld.buf)
        free(ld.data);
}

template <class T>
bool SmallList<T>::empty() const
{
    return ld.num == 0;
}

template <class T>
int SmallList<T>::size() const
{
    return ld.num;
}

template <class T>
T& SmallList<T>::operator[](int n)
{
    assert(n >= 0 && n < ld.num);
    return ld.data[n];
}

template <class T>
const T& SmallList<T>::operator[](int n) const
{
    assert(n >= 0 && n < ld.num);
    return ld.data[n];
}

template <class T>
void SmallList<T>::clear()
{
    ld.num = 0;
}

template <class T>
void SmallList<T>::reserve(int n)
{
    if (n > ld.cap)
    {
        T* new_mem = static_cast<T*>(malloc(n * sizeof *ld.data));
        if (!new_mem)
            throw std::bad_alloc();
         memcpy(new_mem, ld.data, ld.cap * sizeof *new_mem);
        if (ld.data != ld.buf)
            free(ld.data);
        ld.data = new_mem;
        ld.cap = n;
    }
}

template <class T>
void SmallList<T>::resize(int num, const T& fill)
{
    if (num > ld.cap)
        reserve(num + 1);
    if (num > ld.num)
    {
        for (int j = ld.num; j < num; ++j)
            ld.data[j] = fill;
    }
    ld.num = num;
}

template <class T>
void SmallList<T>::push_back(const T& element)
{
    if (ld.num >= ld.cap)
        reserve(ld.cap * 2);
    ld.data[ld.num++] = element;
}

template <class T>
T SmallList<T>::pop_back()
{
    return ld.data[--ld.num];
}

template <class T>
void SmallList<T>::swap(SmallList& other)
{
    ListData& ld1 = ld;
    ListData& ld2 = other.ld;

    const int use_fixed1 = ld1.data == ld1.buf;
    const int use_fixed2 = ld2.data == ld2.buf;

    const ListData temp = ld1;
    ld1 = ld2;
    ld2 = temp;

    if (use_fixed1)
        ld2.data = ld2.buf;
    if (use_fixed2)
        ld1.data = ld1.buf;
}

// ---------------------------------------------------------------------------------
// FreeList Implementation
// ---------------------------------------------------------------------------------
template <class T>
FreeList<T>::FreeList(): first_free(-1)
{
}

template <class T>
int FreeList<T>::insert(const T& element)
{
    if (first_free != -1)
    {
        const int index = first_free;
        first_free = data[first_free].next;
        data[index].element = element;
        return index;
    }
    else
    {
        FreeElement fe;
        fe.element = element;
        data.push_back(fe);
        return data.size() - 1;
    }
}

template <class T>
void FreeList<T>::erase(int n)
{
    assert(n >= 0 && n < data.size());
    data[n].next = first_free;
    first_free = n;
}

template <class T>
void FreeList<T>::clear()
{
    data.clear();
    first_free = -1;
}

template <class T>
int FreeList<T>::range() const
{
    return data.size();
}

template <class T>
T& FreeList<T>::operator[](int n)
{
    return data[n].element;
}

template <class T>
const T& FreeList<T>::operator[](int n) const
{
    return data[n].element;
}