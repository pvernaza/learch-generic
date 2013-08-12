#ifndef __FAST_MARCHING_H__
#define __FAST_MARCHING_H__

#include <queue>
#include <iostream>
#include <assert.h>

typedef enum { ARRAY_C_LAYOUT, ARRAY_FORTRAN_LAYOUT } ArrayNDLayout;

using namespace std;

/**
   Just stores some meta-information for the open list
 */
template <class Elt>
class OpenListElt {
public:
  OpenListElt(int _id, int _age, double _priority, const Elt& _elt) :
    id(_id),
    age(_age), 
    priority(_priority), 
    elt(_elt) {
    /*
    cout << "Creating element with age " << age <<
      " Priority " << priority << endl;
    */
  };

  int id;
  int age;
  double priority;
  Elt elt;
};
 
template <class Elt>
class OpenListEltComparer {
public:
  bool operator()(const OpenListElt<Elt>& e1, const OpenListElt<Elt>& e2) {
    // this makes it a min heap
    return e1.priority > e2.priority;
  }
};

template <class Elt>
class OpenList {
public:
  OpenList(int maxOpen) {
    //    cout << "Making open list, max "  << maxOpen << endl;

    eltAge = new int[maxOpen];
    
    for (int ii = 0; ii < maxOpen; ii++)
      eltAge[ii] = 0;
  };

  virtual ~OpenList() { 
    delete [] eltAge;
  };

  /**
     @param id Identifier, which should be unique for each element added
     @param elt The element being added

     If an element with an identifier already in the queue is added,
     the prior one is ignored.
   */
  void add(int id, double priority, const Elt& elt) {
    if (eltAge[id] < 0)
      eltAge[id] = -eltAge[id];

    eltAge[id]++;

    prioQueue.push(OpenListElt<Elt>(id, eltAge[id], priority, elt));
  }

  /**
     @param id Identifier of the element to be removed
   */
  void remove(int id) {
    // Signify removal by flipping the sign of the age
    // Allows us to add multiple times, then remove, then add again...

    eltAge[id] = -eltAge[id];
  }

  Elt top() {
    OpenListElt<Elt> pqTop = prioQueue.top();

    if (pqTop.age == eltAge[pqTop.id]) {
      return pqTop.elt;
    } else {
      pop();

      return top();
    }
  }

  /**
     Should only be called after top()
   */
  void pop() {
    prioQueue.pop();
    /*
    if (prioQueue.empty())
      return;

    prioQueue.pop();

    OpenListElt<Elt> pqTop = prioQueue.top();

    if (pqTop.age != eltAge[pqTop.id])
      pop();
    */
  }

  bool empty() {
    if (prioQueue.empty())
      return true;

    OpenListElt<Elt> pqTop = prioQueue.top();

    // not empty if there's a valid element on top
    if (pqTop.age == eltAge[pqTop.id]) {
      //      cout << "Empty: No. age = " << pqTop.age << endl; 

      return false;
    } else {
      // discard invalid elements
      pop();
      
      return empty();
    }
  }

private:
  int *eltAge;			// array of element "ages"
  priority_queue<OpenListElt<Elt>, 
		 vector<OpenListElt<Elt> >, 
		 OpenListEltComparer<Elt> > prioQueue;
};

/**
   Thin wrapper class that provides accessors to a bare ND array
 */
template <class Elt>
class ArrayND {
public:
  ArrayND(Elt* _costarray, const vector<int>& _dims, ArrayNDLayout layout) : 
    costarray(_costarray),
    dims(_dims)
  {
    if (layout == ARRAY_C_LAYOUT) {
      iStart = dims.size() - 1;
      iEnd = 0;
      iDelta = -1;
    } else if (layout == ARRAY_FORTRAN_LAYOUT) {
      iStart = 0;
      iEnd = dims.size() - 1;
      iDelta = 1;
    } else {
      assert(false);
    }
  }

  virtual ~ArrayND() { };

  Elt get(int index) const {
    return costarray[index];
  }

  Elt get(const vector<int>& subs) const {
    return costarray[sub2ind(subs)];
  }
  
  void set(int index, const Elt& value) {
    costarray[index] = value;
  }

  void set(const vector<int>& subs, const Elt& value) {
    costarray[sub2ind(subs)] = value;
  }
  
  int numel() const {
    int numel = 1;
    for (unsigned int ii = 0; ii < dims.size(); ii++) {
      numel *= dims[ii];
    }
    return numel;
  }

  void ind2sub(int ind, vector<int>& subs) const {
    subs.resize(ndims(), 0);
    ind2sub(dims, ind, subs, iStart, iEnd, iDelta);
  }

  int sub2ind(const vector<int>& subs) const {
    return sub2ind(dims, subs, iStart, iEnd, iDelta);
  }

  // Assumes first subscript stored consecutively in memory
  // for c layout: xi = 0, xend = size subs - 1, delta = 1
  // for fortran layout: xi = size subs - 1, xend = 0, delta = -1
  static int sub2ind(const vector<int>& dims, 
		     const vector<int>& subs, 
		     unsigned int xi, 
		     unsigned int xend,
		     int delta) {

    if (xi == xend) return subs[xi];

    return subs[xi] + dims[xi] * sub2ind(dims, subs, xi+delta, xend, delta);
  }

  /*
    NB: precondition: subs.size() == dims()
   */
  static void ind2sub(const vector<int>& dims,
		      int ind, 
		      vector<int>& subs,
		      unsigned int xi,
		      unsigned int xend,
		      int delta) {
    
    int sub = ind % dims[xi];

    subs[xi] = sub;

    if (xi == xend) return;
    
    ind2sub(dims, (ind - sub) / dims[xi], subs, xi+delta, xend, delta);
  }

  unsigned int ndims() const {
    return dims.size();
  }

  vector<int> getDims() const {
    return dims;
  }

private:
  
  Elt *costarray;
  vector<int> dims;

  unsigned int iStart, iEnd;
  int iDelta;
};

class LatticePoint {
public:
  LatticePoint(int _id, int _nbDim) : 
    id(_id),
    nbDim(_nbDim)
  {
  }

  int id;
  int nbDim;			
};

class SetCompareLatticePoint {
public:
  bool operator()(const LatticePoint& p1, const LatticePoint& p2) const {
    return p1.id < p2.id;
  }
};

template <class Elt>
class Predicate {
public:
  virtual bool isTrue(const Elt&) const = 0;
};

/*
bool fastMarching(const ArrayND<double>& costmap, 
		  const vector<int> goal,
		  ArrayND<double>& valf);
*/

/**
   \param costmap A grid of costs
   \param costDx The dimensions of each grid cell
   \param goal The goal. If empty, all cells with negative costs are assumed to 
               be goal states.
   \param stop If nonempty, the calculation stops when this cell is expanded
   \param valf Filled with value function on return
 */
bool fastMarching(ArrayND<double>& costmap, 
		  const vector<double>& costDx,
		  const vector<int> goal,
                  const vector<int> stop,
		  ArrayND<double>& valf);


#endif
