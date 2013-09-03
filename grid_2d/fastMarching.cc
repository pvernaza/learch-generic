#include <algorithm>
#include <assert.h>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fastMarching.hh"

//extern "C" {
//#include "fastMarchingC.h"
//}

typedef set<LatticePoint, SetCompareLatticePoint> LatticePointSet;

class BurntPredicate : public Predicate<LatticePoint> {
public:
  BurntPredicate(const LatticePointSet* _burnt) : burnt(_burnt) {};

  bool isTrue(const LatticePoint& point) const {
    LatticePointSet::const_iterator it = burnt->find(point);
    bool value = (it != burnt->end());

    //    cout << "Point " << point.id << " " << value << endl;

    return value;
  }

private:
  const LatticePointSet* burnt;
};

class UnburntPredicate : public Predicate<LatticePoint> {
public:
  UnburntPredicate(LatticePointSet* _burnt) : burnt(_burnt) {};

  bool isTrue(const LatticePoint& point) const {
    LatticePointSet::const_iterator it = burnt->find(point);
    bool value = (it == burnt->end());

    //    cout << "Point " << point.id << " " << value << endl;

    return value;
  }

  private:
    LatticePointSet* burnt;
};

void getNeighbors(const ArrayND<double>& costmap, 
		  const ArrayND<double>& valf,
		  const int latticeId, 
		  const Predicate<LatticePoint>& predicate,
		  LatticePointSet& neighbors,
		  bool rejectLarger = false);

void updateValues(const ArrayND<double>& costmap, 
		  const vector<double>& costDx,
		  const LatticePointSet& burnt, 
		  const LatticePointSet& latticePoints, 
		  ArrayND<double>& valf);

void testArrayND();

void printCoords(const vector<int>& coords) {
  cout << "[ ";

  for (unsigned int ii = 0; ii < coords.size(); ii++)
    cout << coords[ii] << " ";
  
  cout << "] ";
}

void printPointSet(const LatticePointSet& pointSet) {
  cout << "[ ";
  for (LatticePointSet::const_iterator it = pointSet.begin();
       it != pointSet.end(); 
       it++) {
    cout << (*it).id << " " ;
  }
  cout << "] " << endl;
}

/**
   FIXME: now initializing valf to infinity -- otherwise, 
   I could possibly get incorrect values where input cost is infinite.
   Does this break anything?
 */
/**
   If no goal is specified, any cells with negative cost are assumed 
   to be goals.  The costs of such cells are then negated to make 
   them positive.

   If stop vector is non-zero length, calculation stops when 
   stop is expanded.
 */
// REMEMBER TO USE RETURN VALUE
bool fastMarching(ArrayND<double>& costmap, 
		  const vector<double>& costDx,
		  const vector<int> goal,
                  const vector<int> stop,
		  ArrayND<double>& valf) {

  int nLatticePoints = costmap.numel();

  LatticePointSet burnt;
  OpenList<LatticePoint> burning(nLatticePoints);

  for (int ii = 0; ii < valf.numel(); ii++) 
    valf.set(ii,INFINITY);
  
  if (goal.size() > 0) {
    if (!isfinite(costmap.get(goal))) {
      cerr << "Goal has non-finite cost. Exiting." << endl;
      return false;
    }

    int goalIndex = costmap.sub2ind(goal);
    burning.add(goalIndex, 0, LatticePoint(goalIndex, -1));
    valf.set(goalIndex,0);
  } else {                      // negative-cost cells assumed to be goals
    for (int ii = 0; ii < costmap.numel(); ii++) {
      if (costmap.get(ii) <= 0) {
        costmap.set(ii, -costmap.get(ii));
        valf.set(ii, 0);
        burning.add(ii, 0, LatticePoint(ii, -1));
        // need to add to burnt to avoid re-processing goal nodes
        burnt.insert(LatticePoint(ii,-1));
      }
    }
  }

  int stopIndex = -1;
  if (stop.size() > 0) {
    stopIndex = costmap.sub2ind(stop);
  }

  int nBurnt = 0;

  while (!burning.empty()) {
    LatticePoint activePoint = burning.top();
    burning.pop();

    if (activePoint.id == stopIndex) {
      printf("FMM reached stopping point\n");
      break;
    }

    /*
    cout << "Active: " << activePoint.id << " Value: " << 
      valf.get(activePoint.id) << endl;
    */
    //    printf("S %d = %f\n", activePoint.id, valf.get(activePoint.id));


    burnt.insert(activePoint);

    LatticePointSet unburntNeighbors;
    getNeighbors(costmap, valf, activePoint.id, 
		 UnburntPredicate(&burnt), unburntNeighbors);

    updateValues(costmap, costDx, burnt, unburntNeighbors, valf);

    // add updated neighbors to burning list
    for (LatticePointSet::const_iterator it = unburntNeighbors.begin();
	 it != unburntNeighbors.end();
	 it++) {

      int index = (*it).id;
      burning.add(index, valf.get(index), LatticePoint(index, -1));
    }

    nBurnt++;

    if ((nBurnt % 10000) == 0)
      cout << "Progress: " << (double)nBurnt / costmap.numel() << endl;
  }

  //  printf("Iterations = %d\n", nBurnt);
  fflush(stdout);

  return true;
}

void updateValues(const ArrayND<double>& costmap, 
		  const vector<double>& costDx,
		  const LatticePointSet& burnt, 
		  const LatticePointSet& latticePoints, 
		  ArrayND<double>& valf) {

  for (LatticePointSet::const_iterator it = latticePoints.begin();
       it != latticePoints.end();
       it++) {
    
    // the lattice point whose value will be updated
    LatticePoint currentPoint = *it;

    //    cout << "Updating " << currentPoint.id << endl;
    
    LatticePointSet burntNeighbors;
    getNeighbors(costmap, valf, currentPoint.id, 
		 BurntPredicate(&burnt), burntNeighbors, true);
    
    // calculate quadratic coefficients
    double pcoeff[3];
    memset(pcoeff, 0, sizeof(double)*3);

    assert(burntNeighbors.size() > 0);
    
    double maxNborValue = -INFINITY;
    double maxNborDelta = 0.0;

    for (LatticePointSet::const_iterator nborIt = burntNeighbors.begin();
	 nborIt != burntNeighbors.end();
	 nborIt++) {

      LatticePoint neighbor = *nborIt;

      //      cout << "Neighbor " << neighbor.id << endl;

      double value = valf.get(neighbor.id);
      double delta = costDx[neighbor.nbDim];
      
      if (value > maxNborValue) { 
        maxNborValue = value;
        maxNborDelta = delta;
      }
      
      //      printf("V %f D %f\n", value, delta);

      pcoeff[0] += (1 / (delta*delta));
      pcoeff[1] += -2*value / (delta*delta);
      pcoeff[2] += ((value * value) / (delta * delta));
    }

    double cost = costmap.get(currentPoint.id);
    pcoeff[2] -= cost*cost;

    //    printf("Coeffs = [ %f, %f, %f ]\n", pcoeff[0], pcoeff[1], pcoeff[2]);

    double discrim = pcoeff[1]*pcoeff[1] - 4*pcoeff[0]*pcoeff[2];
    
    /*
    assert(discrim >= 0);
    */

    if (discrim < 0) {
      fprintf(stderr, "Discrim = %f\n", discrim);
      discrim = 0;
    }
 
    double quadRoot = (-pcoeff[1] + sqrt(discrim)) / (2 * pcoeff[0]);

    double solution = 0.0;
    // FIXME: hack!!
    // if value is invalid, pretend we came 
    // straight from the neighbor with maximum cost
    if (quadRoot <= maxNborValue) {
      fprintf(stderr, "est.val. = %f <= %f, HACKING IT!!\n",
              quadRoot, maxNborValue);
      solution = maxNborValue + maxNborDelta * cost;
      if (solution <= maxNborValue) {
        fprintf(stderr, "solution = %f, maxNborValue = %f, maxNborDelta = %f, cost = %f\n", solution, maxNborValue, maxNborDelta, cost);
        // this actually does fail sometimes...
        //        assert(solution > maxNborValue);
      }
    } else {
      solution = quadRoot;
    }
    
    assert(!isnan(solution));
    assert(solution >= 0);

    valf.set(currentPoint.id, solution);
  }
}

/**
   @param rejectLarger If true, if there are two neighbors in a given 
   dimension, does not return the neighbor with the larger value
 */
void
getNeighbors(const ArrayND<double>& costmap, 
	     const ArrayND<double>& valf,
	     const int latticeId, 
	     const Predicate<LatticePoint>& predicate,
	     LatticePointSet& neighbors,
	     bool rejectLarger) {
  
  vector<int> pointSubs;
  costmap.ind2sub(latticeId, pointSubs);

  vector<int> dims = costmap.getDims();
 
  for (unsigned int dim = 0; dim < costmap.ndims(); dim++) {
    vector<int> subsDown = pointSubs;
    vector<int> subsUp = pointSubs;

    subsDown[dim] -= 1;
    subsUp[dim] += 1;

    //    vector<LatticePoint> tentative;
    int indexDown = costmap.sub2ind(subsDown);
    LatticePoint pointDown(indexDown, dim);

    int indexUp = costmap.sub2ind(subsUp);
    LatticePoint pointUp(indexUp, dim);

    vector<LatticePoint> tentative;

    // check array bounds, predicate, and finiteness
    // FIXME: finiteness check necessary?
    if (subsDown[dim] >= 0 && 
	predicate.isTrue(pointDown) &&
	isfinite(costmap.get(indexDown))) {
    //    if (subsDown[dim] >= 0) {
      tentative.push_back(pointDown);
    }
      
    if (subsUp[dim] < dims[dim] && 
	predicate.isTrue(pointUp) &&
	isfinite(costmap.get(indexUp))) {
    //    if (subsUp[dim] < dims[dim]) {
      tentative.push_back(pointUp);
    }

    if (rejectLarger && tentative.size() == 2) {
      if (valf.get(tentative[0].id) < valf.get(tentative[1].id))
	tentative.erase(tentative.begin() + 1);
      else
	tentative.erase(tentative.begin());
    }

    neighbors.insert(tentative.begin(), tentative.end());
  }
}

/*
//extern "C"
int fastMarchingC(double *costmap,
                  const int *dims,
                  const int *goal,
                  int nDims,
                  double *valf) {

  vector<int> dimvec, goalvec;

  for (int ii = 0; ii < nDims; ii++) {
    dimvec.push_back(dims[ii]);
    goalvec.push_back(goal[ii]);
  }

  ArrayND<double> cost(costmap, dimvec);
  ArrayND<double> valfunc(valf, dimvec);

  bool result = fastMarching(cost, goalvec, valfunc);

  return result ? 1 : 0;
}
*/

void testArrayND() {
  vector<int> dims;
  dims.push_back(50);
  dims.push_back(100);
  dims.push_back(200);

  double costarray[dims[0] * dims[1] * dims[2]];

  ArrayND<double> costmap(costarray, dims, ARRAY_C_LAYOUT);

  for (int ii = 0; ii < 10; ii++) {
    vector<int> subs;
    subs.push_back(floor(drand48() * dims[0]));
    subs.push_back(floor(drand48() * dims[1]));
    subs.push_back(floor(drand48() * dims[2]));
    double val = drand48();

    costmap.set(subs, val);

    printf("[ %d %d %d ] -> %d\n", subs[0], subs[1], subs[2], 
	   costmap.sub2ind(subs));

    vector<int> foundSubs;
    costmap.ind2sub(costmap.sub2ind(subs), foundSubs);

    printf("%d -> [ %d %d %d ]\n", 
	   costmap.sub2ind(subs),
	   foundSubs[0], foundSubs[1], foundSubs[2]);

    printf("Set value %f, retrieved %f\n", val, costmap.get(subs));
  }
}


void testOpenList() {
  OpenList<string> olist(10000);

  olist.add(1, 100, string("1"));

  cout << olist.top() << endl;

  olist.add(2, 200, string("2"));

  cout << olist.top() << endl;

  olist.add(3, 90, string("3"));

  cout << olist.top() << endl;

  olist.add(3, 91, string("3"));

  cout << olist.top() << endl;

  olist.remove(3);

  cout << olist.top() << endl;
  
  olist.add(4, 100, string("4"));

  cout << olist.top() << endl;

  olist.add(4, 1, string("4"));

  cout << olist.top() << endl;

  olist.add(3, -1, string("3"));

  cout << olist.top() << endl;

  while (!olist.empty()) {
    cout << "Pop: " << olist.top() << endl;
    olist.pop();
  }
}


/*
int main(int argc, char *argv[]) {
  testArrayND();

  exit(0);

  vector<int> dims;
  dims.push_back(50);
  dims.push_back(50);
  dims.push_back(50);
  //  dims.push_back(200);

  double* costarray = new double [dims[0] * dims[1] * dims[2]];
  double* valfarray = new double [dims[0] * dims[1] * dims[2]];

  ArrayND<double> costmap(costarray, dims, ARRAY_C_LAYOUT); 
  ArrayND<double> valf(valfarray, dims, ARRAY_C_LAYOUT); 

  for (int ii = 0; ii < costmap.numel(); ii++)
    costarray[ii] = 1;

  vector<int> goal;
  goal.push_back(10);
  goal.push_back(10);
  goal.push_back(10);
  //  goal.push_back(1);

  vector<double> dx;
  dx.push_back(1);
  dx.push_back(1);
  dx.push_back(1);
  //  dx.push_back(1);

  fastMarching(costmap, dx, goal, valf);

  FILE* file = fopen("valueFunction.dat", "w");
  assert(file != NULL);
  for (int ii = 0; ii < costmap.numel(); ii++) 
    fprintf(file, "%f ", valf.get(ii));

  delete [] costarray;
  delete [] valfarray;
}
*/

