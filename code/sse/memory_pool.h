#ifndef _MEMORY_POOL_H_
#define _MEMORY_POOL_H_

#include <vector>

template<class T>
struct memory_pool {


  std::vector<T> vec;
  size_t capacity;
  size_t used;


  memory_pool() {
    vec.clear();
    capacity = 0;
    used = 0;
  };
  ~memory_pool() {};
  
  /////////
  
  T *allocate(size_t ss) {
    if (ss > capacity) {
      vec.resize(ss);
      capacity = ss;
    }
    used = 0;
    return &vec[0];
  }

  T *get(const int count) {
    T *ret = &vec[used];
    used += count;
    assert(used <= capacity);
    
    return ret;
  };

  T& operator[](int i){return vec[i];}
  void reset() {
    used = 0;
  }

};


#endif // _MEMORY_POOL_H_
