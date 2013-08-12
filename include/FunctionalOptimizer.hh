#ifndef __FUNCTIONAL_OPTIMIZER_H__
#define __FUNCTIONAL_OPTIMIZER_H__

#include "ObjectiveFunctional.hh"
#include "Regressor.hh"

/**
   This class defines operations supported by a functional optimizer.
   \tparam FunclOpt The type of a functional optimizer
   \tparam ObjFuncl The type of an objective functional
   \tparam Reg The type of a regressor (i.e., the type of function to optimize)
   \tparam RegParams The type of parameters that control the regressor training
 */
template <typename FunclOpt, typename ObjFuncl, 
          typename RegParams, typename Reg>
class FunctionalOptimizerOps {

public:

  /**
     Optimize the given objective functional with the given optimizer, 
     starting at the initial function reg0.
   */
  static Reg
  Optimize(const FunclOpt& funclOpt,
           const ObjFuncl& objFuncl, 
           const RegParams& regParams,
           const Reg& reg0, 
           int nIterations);

};

/**
   This class represents a naive functional optimizer, implemented
   by taking approximate gradient steps using the given regressor.
 */
struct NaiveFunctionalOptimizer {
  explicit NaiveFunctionalOptimizer(double _learnRate)
    : learnRate(_learnRate) {}
  double learnRate;
};

/**
   Implementation of operations on NaiveFunctionalOptimizer
 */
template <typename ObjFuncl, typename RegParams, typename Reg>
class FunctionalOptimizerOps<NaiveFunctionalOptimizer,
                             ObjFuncl, 
                             RegParams,
                             Reg> {
  
public:

  typedef NaiveFunctionalOptimizer FunclOpt;
  typedef ObjectiveFunctionalOps<ObjFuncl, Reg> ObjFunclOps;

  static Reg
  Optimize(const FunclOpt& funclOpt,
           const ObjFuncl& objFuncl,
           const RegParams& regParams,
           const Reg& reg0,
           int nIterations) {
    if (nIterations == 0) return reg0;
    std::vector<std::pair<learch_vector, double> > trainData = ObjFunclOps::Gradient(objFuncl, reg0);
    Reg reg = RegressorParamOps<RegParams,Reg>::Train(regParams, trainData);
    reg = ComposedRegressorOps<Reg>::Scale(reg, -funclOpt.learnRate);
    reg = ComposedRegressorOps<Reg>::Add(reg, reg0);
    reg = ObjFunclOps::ProjectFeasibleSet(objFuncl, reg);
    return Optimize(funclOpt, objFuncl, regParams, reg, nIterations - 1);
  }

};

/**
   Repeated gradient projection algorithm from
   "Grubb, Bagnell. Generalized boosting algorithms for convex optimization"
 */
class RepeatedGradientFunctionalOptimizer {
  
public:
  
  explicit RepeatedGradientFunctionalOptimizer(double _learnRate) 
    : learnRate(_learnRate) {}
  
  double learnRate;
};

/**
   Implementation of repeated gradient projection algorithm
 */
template <typename ObjFuncl, typename RegParams, typename Reg>
class FunctionalOptimizerOps<RepeatedGradientFunctionalOptimizer,
                             ObjFuncl,
                             RegParams,
                             Reg> {

public:

  typedef RepeatedGradientFunctionalOptimizer FunclOpt;
  typedef ObjectiveFunctionalOps<ObjFuncl,Reg> ObjFunclOps;
  typedef std::pair<learch_vector, double> ScaledShiftedSpike;
  typedef std::vector<ScaledShiftedSpike> ScaledShiftedSpikes;

  static Reg
  Optimize(const FunclOpt& funclOpt,
           const ObjFuncl& objFuncl,
           const RegParams& regParams,
           const Reg& reg0,
           int nIterations) {
    return Optimize(funclOpt, objFuncl, regParams, reg0, nIterations, 1);
  }

private:

  static Reg
  Optimize(const FunclOpt& funclOpt,
           const ObjFuncl& objFuncl,
           const RegParams& regParams,
           const Reg& reg0,
           int nIterations,
           int nGradBoostRounds) {
    ScaledShiftedSpikes grad0 = ObjFunclOps::Gradient(objFuncl, reg0);




    Reg approxGrad = RegressorParamOps<RegParams,Reg>::Train(regParams, grad0);
    approxGrad = ApproxGradient(approxGrad, regParams, grad0, nGradBoostRounds);





    approxGrad = ComposedRegressorOps<Reg>
      ::Scale(approxGrad, -funclOpt.learnRate);
    Reg reg = ComposedRegressorOps<Reg>::Add(reg0, approxGrad);
    reg = ObjFunclOps::ProjectFeasibleSet(objFuncl, reg);
    return Optimize(funclOpt, objFuncl, regParams, reg, 
                    nIterations - 1, nGradBoostRounds + 1);
  }

  static Reg
  ApproxGradient(const Reg& approxGrad0,
                 const RegParams& regParams,
                 const ScaledShiftedSpikes& residGrad,
                 int nIterations) {
    if (nIterations <= 1) return approxGrad0;

    // approximate the remaining part of the functional gradient
    Reg approxGrad = RegressorParamOps<RegParams,Reg>
      ::Train(regParams, residGrad);

    // FIXME: I think this part is slightly dubious, theoretically.
    // We shouldn't just care about the residual where the examples are--
    // we should also care about the residual where the examples /are not/.
    ScaledShiftedSpikes newResidGrad = 
      ComputeFunctionalResidual(approxGrad, residGrad);

    // FIXME: assuming no more scaling is needed to maximize projection
    Reg reg = ComposedRegressorOps<Reg>::Add(approxGrad0, approxGrad);
    return ApproxGradient(reg, regParams, newResidGrad, nIterations - 1);
  }

  // FIXME: again, this is not the true functional residual
  static ScaledShiftedSpikes
  ComputeFunctionalResidual(const Reg& reg,
                            const ScaledShiftedSpikes& target) {
    ScaledShiftedSpikes residual;
    for (int ii = 0; ii < target.size(); ii++) {
      const learch_vector& state = target[ii].first;
      const double output = target[ii].second;
      double predicted = BasicRegressorOps<Reg>::Eval(reg, state);
      double resid = output - predicted;
      residual.push_back(ScaledShiftedSpike(state, resid));
    }
    return residual;
  }
};


/**
   Residual gradient method from 
   "Grubb, Bagnell. Generalized boosting algorithms for convex optimization"
 */
/*
class ResidGradientFunctionalOptimizer {
  
public:

  friend class FunctionalOptimizerOps<ResidGradientFunctionalOptimizer, 
                                      ObjFuncl, 
                                      RegParams,
                                      Reg>;
  
  explicit ResidGradientFunctionalOptimizer(double _learnRate)
    : learnRate(_learnRate) {}

  double learnRate;
};
*/

/**
   Implementation of residual gradient method

   Problem: in this method, residual is not a sum of delta functions, 
   since it includes elements from the hypothesis class.  This breaks 
   the current interface
 */

/*
template <typename ObjFuncl, typename Reg>
class FunctionalOptimizerOps<ResidGradientFunctionalOptimizer<ObjFuncl,Reg>,
                             ObjFuncl,
                             Reg> {

public:

  typedef ResidGradientFunctionalOptimizer<ObjFuncl,Reg> FunclOpt;
  typedef ObjectiveFunctionalOps<ObjFuncl,Reg> ObjFunclOps;
  typedef std::pair<learch_vector, double> ScaledShiftedSpike;
  typedef std::vector<ScaledShiftedSpike> ScaledShiftedSpikes;

  static Reg
  Optimize(const FunclOpt& funclOpt,
           const ObjFuncl& objFuncl,
           const Reg& reg0,
           int nIterations,
           const ScaledShiftedSpikes& resid = ScaledShiftedSpikes()) {
    if (nIterations == 0) return reg0;
    
  }

private:

  ScaledShiftedSpikes
  AddSpikes(const ScaledShiftedSpike& s0, 
            const ScaledShiftedSpike& s1) {
    ScaledShiftedSpikes result;

  }


};
*/

#endif
