#ifndef _BINARY_TREE_H_
#define _BINARY_TREE_H_

template<class T>
struct binary_tree {
  T value;
  binary_tree *left, *right;
  binary_tree() {
    left = right = NULL;
  }
  ~binary_tree() {clear();}
  bool isempty() {return left == NULL;}

  void clear() {
    if (left  != NULL) delete left;
    if (right != NULL) delete right;
    left = right = NULL;
  }
  
  bool insert(const T v) {
    if (isempty()) {
      value = v;
      left  = new binary_tree;
      right = new binary_tree;
      return true;
    } else {
      if       (v < value) return left ->insert(v);
      else if  (v > value) return right->insert(v);
      else                 return false;
    }
  }

  bool isintree(const T v) {
    if (isempty()) return false;
    
    if      (v < value) return left ->isintree(v);
    else if (v > value) return right->isintree(v);
    else                return true;
    
  }

  void dump(FILE *fout, const char line[] = "%d  ") {
    if (isempty()) return;

    left ->dump(fout, line);
    right->dump(fout, line);
    
    fprintf(fout, line, value);
  }
};

#endif // _BINARY_TREE_H_
