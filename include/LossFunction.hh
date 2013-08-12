#ifndef __LOSS_FUNCTION_H__
#define __LOSS_FUNCTION_H__

#include <math.h>

/**
   Interface for defining loss function.
   The actual loss function (defined on paths) is the sum of this 
   stagewise loss function over the path.
 */
template <typename LossFunc>
class LossFunctionOps {
  
public:
  
  static double 
  Eval(const LossFunc& lossFunc, 
       const learch_path& refPath,
       const learch_vector& state);

  static LossFunc 
  ZeroLossFunc();

};

/**
   Loss function described in Section 6 of Auton. Robot. paper
 */
class ExpMinDistToPathLoss {
  
  friend class LossFunctionOps<ExpMinDistToPathLoss>;

public:
  
  ExpMinDistToPathLoss(double _sigma, bool _isZero = false) 
    : sigma(_sigma), isZero(_isZero) {
  }

private:

  double sigma;
  bool isZero;

};

/**
   Implementation of ExpMinDistToPathLoss loss function
 */
template<>
class LossFunctionOps<ExpMinDistToPathLoss> {
  
public:

  static double 
  Eval(const ExpMinDistToPathLoss& loss,
       const learch_path& refPath,
       const learch_vector& state) {
    if (loss.isZero) { 
      return 0;
    } else {
      double minDist = INFINITY;
      for (int iPath = 0; iPath < refPath.size(); iPath++) {
        const learch_vector refState = refPath[iPath];
        double dist = (state - refState).norm();
        if (dist < minDist) minDist = dist;
      }
      double pathLen = refPath.size();
      return ((1 - (exp (- (minDist*minDist) / (loss.sigma*loss.sigma)))) / pathLen);
    }
  }

  static ExpMinDistToPathLoss
  ZeroLossFunc() {
    return ExpMinDistToPathLoss(0., true);
  }

};

#endif
