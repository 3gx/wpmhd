#ifndef _BINARYNODE_H_
#define _BINARYNODE_H_

#include "memory_pool.h"

template<class T>
struct binarynode {
  T value;
  binarynode *next;
  binarynode() {clear();}
  ~binarynode() {};
  bool isempty() {return next == NULL;}
  
  void set(const T v, memory_pool<binarynode> &pool) {
    value = v;
    next = pool.get(2);
    next[0].next = next[1].next = NULL;
  };
  void clear() {
    next = NULL;
  }
  
  T get() const {return value;}

  static bool insert(binarynode &node, const T v, memory_pool<binarynode> &pool) {
    binarynode *current = &node;
    while (true) {
      if (current->isempty()) {
	current->set(v, pool);
	return true;
      } else {
	if       (v < current->get()) current = &current->next[0];    
	else if  (v > current->get()) current = &current->next[1];
	else                          return false;
      }
    }
    return false;
  }

  bool isintree(const T v) {
    if (isempty()) return false;
    
    binarynode *current = this;
    if      (v < value) current = &current->next[0];
    else if (v > value) current = &current->next[1];
    else                return true;
    
    return current->isintree(v);
  }

  void dump(FILE *fout, const char line[] = "%d  ") {
    if (isempty()) return;

    binarynode *current = this;
    current->next[0].dump(fout, line);
    current->next[1].dump(fout, line);
    
    fprintf(fout, line, value);
  }
};

#endif // _BINARYNODE_H_
