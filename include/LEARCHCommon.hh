#ifndef __LEARCH_COMMON_H__
#define __LEARCH_COMMON_H__

#include <Eigen/Dense>
#include <numeric>
#include <vector>

typedef Eigen::VectorXd learch_vector;

/**
   Path is represented by a vector of states, represented as 
   learch_vectors.
 */
typedef std::vector<learch_vector> learch_path;

template<typename A, typename B>
std::vector<B> map_vector(B (*func)(const A&), const A& vec) {
  std::vector<B> result = new std::vector<B>(vec.size());
  std::transform(vec.start(), vec.end(), result->start(), func);
  return result;
}

template<typename Acc, typename Elt>
Acc fold_vector(Acc (*func)(const Acc&, const Elt&), 
                const Acc& acc,
                const std::vector<Elt> elts) {
  std::accumulate(elts.start(), elts.end(), acc, func);
}

#endif
