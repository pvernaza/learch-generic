#ifndef __LEARCH_OBJECTIVE_H__
#define __LEARCH_OBJECTIVE_H__

#include "LEARCHCommon.hh"
#include "LEARCHPlanner.hh"
#include "LossFunction.hh"
#include "ObjectiveFunctional.hh"
#include "Regressor.hh"

#include <float.h>

/**
   The LEARCHObjective class is parameterized over the implementations
   of several key operations. The template parameters of the class
   represent the abstract types to which these operations apply.  For
   instance, the operation of feature function evaluation is applied
   to a feature function and an environment to produce a cost, and is
   hence parameterized over the types of feature functions and
   environments. Similarly, LEARCHObjective is parameterized over the
   types of the environment (Env) and the type (LossFunc) of the loss
   function, among other things.

   The operations defined on each of these types are specified by a
   class templated over the type.  For instance, operations involving
   a loss function of type LossFunc are specified by the class
   LossFunctionOps<LossFunc>.  The implementations of these classes
   are provided via template specialization; i.e.,, once the user
   decides on a concrete type for LossFunc, he will implement
   operations on LossFunc by providing a specialization of the class
   LossFunctionOps for this concrete type.
 */

/**
   The loss-augmented cost function to pass to the planner
 */
template <typename Env, typename Reg, 
          typename FeatureFunc, typename LossFunc>
class LossAugPlannerCost : public LEARCHPlannerCost<Env> {
  
public: 

  LossAugPlannerCost(const Reg& _featureCost,
                     const FeatureFunc& _featureFunc,
                     const LossFunc& _lossFunc, 
                     const learch_path& _refPath) 
    : featureCost(_featureCost), featureFunc(_featureFunc), 
      lossFunc(_lossFunc), refPath(_refPath) { }

  double 
  Eval(const Env& environment,
       const learch_vector& state) const {
    learch_vector fVec = 
      FeatureFunction<Env,FeatureFunc>::Eval(featureFunc, environment, state);
    double cost = ComposedRegressorOps<Reg>::Eval(featureCost, fVec);
    double loss = LossFunctionOps<LossFunc>::Eval(lossFunc, refPath, state);
    if ((cost - loss) <= 0) 
      return FLT_MIN;
    return cost - loss;
  }

private:

  const Reg featureCost;
  const FeatureFunc featureFunc;
  const LossFunc lossFunc;
  const learch_path refPath;

};


/**
   The concrete type of the LEARCH objective functional
 */
template <typename Env, typename Reg,
          typename FeatureFunc, typename LossFunc>
class LEARCHObjective {
  
public:

  typedef LEARCHObjective<Env,Reg,FeatureFunc,LossFunc> ObjFuncl;
  friend class ObjectiveFunctionalOps <ObjFuncl, Reg>;

  typedef std::pair<Env, learch_path> env_path;
  typedef std::vector<env_path> env_paths;
  typedef LossAugPlannerCost<Env, Reg, FeatureFunc, LossFunc> 
             _LossAugPlannerCost;

  LEARCHObjective(const env_paths& _trainPaths,
                  const FeatureFunc& _featureFunc, 
                  const LossFunc& _lossFunc) 
    : trainPaths(_trainPaths), featureFunc(_featureFunc), lossFunc(_lossFunc) {
  }

private:

  /**
     Transforms a (environment,path) pair into a path of 
     features along the path. Caller must deallocate result.
   */
  learch_path*
  GetFeaturesAlongPath(const env_path& env_path) const {
    const Env& environment = env_path.first;
    const learch_path& statePath = env_path.second;
    learch_path* featurePath = new learch_path(statePath.size());
    for (int iPath = 0; iPath < statePath.size(); iPath++) {
      (*featurePath)[iPath] = 
        FeatureFunction<Env,FeatureFunc>::Eval
        (featureFunc, environment, statePath[iPath]);
    }
    return featurePath;
  }

  /**
     Concatenates all the features corresponding to the input 
     paths into a single path of features.
  */
  learch_path*
  GetFeaturesAlongPaths(const env_paths& inPaths) const {
    learch_path* result = new learch_path();
    for (typename env_paths::const_iterator iPaths = inPaths.begin();
         iPaths != inPaths.end();
         iPaths++) {
      learch_path* mappedPath = GetFeaturesAlongPath(*iPaths);
      result->insert(result->end(), mappedPath->begin(), mappedPath->end());
      delete mappedPath;
    }
    return result;
  }

  /**
     Returns the (environment and the) path planned by the planner
     with the given cost function (loss-augmented), given the
     environment and start/end points corresponding to the given
     training example.

     \param featureCost A map from features to costs
     \param lossFunc A loss function
     \param inPath A training environment and path pair
   */
  env_path*
  GetPlanForTrainPath(const Reg& featureCost,
                      const LossFunc& lossFunc,
                      const env_path& inPath) const {
    const Env& environment = inPath.first;
    const learch_path& trainPath = inPath.second;
    const learch_vector& start = trainPath[0];
    const learch_vector& goal = trainPath[trainPath.size()-1];
    _LossAugPlannerCost lossAugCost
      (featureCost, featureFunc, lossFunc, trainPath);
    learch_path* path = 
      LEARCHPlanner<Env>::Plan(environment, lossAugCost, start, goal);
    env_path* result = new env_path(environment, *path);
    delete path;
    return result;
  }

  env_paths*
  GetPlansForTrainPaths(const Reg& featureCost,
                        const LossFunc& lossFunc,
                        const env_paths& inPaths) const {
    env_paths* outPaths = new env_paths();
    for (int iPath = 0; iPath < inPaths.size(); iPath++) {
      env_path* path = 
        GetPlanForTrainPath(featureCost, lossFunc, inPaths[iPath]);
      outPaths->push_back(*path);
      delete path;
    }
    return outPaths;
  }

private:

  const env_paths& trainPaths;
  const LossFunc lossFunc;
  const FeatureFunc featureFunc;

};

/**
   Implementation of objective functional operations for LEARCHObjective
 */
template <typename Env, typename Reg,
          typename FeatureFunc, typename LossFunc>
class ObjectiveFunctionalOps
           <LEARCHObjective<Env, Reg, FeatureFunc, LossFunc>, 
            Reg> {

public: 

  typedef LEARCHObjective<Env, Reg, FeatureFunc, LossFunc> ObjFuncl;
  //  typedef ObjectiveFunctionalOps<_LEARCHObjective, Reg> ObjFunclOps;
  // FIXME: why doesn't the following work?
  //  typedef typename _ObjectiveFunctionalOps::ScaledShiftedSpike _ScaledShiftedSpike;
  typedef std::pair<learch_vector, double> _ScaledShiftedSpike;

  static std::vector<_ScaledShiftedSpike>
  Gradient(const ObjFuncl& obj, const Reg& featureCost0) {
    typename ObjFuncl::env_paths *predPaths = 
      obj.GetPlansForTrainPaths(featureCost0, obj.lossFunc, obj.trainPaths);

    learch_path* trainFeatures = obj.GetFeaturesAlongPaths(obj.trainPaths);
    learch_path* predFeatures = obj.GetFeaturesAlongPaths(*predPaths);
      
    std::vector<_ScaledShiftedSpike> gradient = 
      FunclGradientFromFeatures(*trainFeatures, *predFeatures);

    delete predPaths;
    delete trainFeatures;
    delete predFeatures;

    return gradient;
  }

  static Reg
  ProjectFeasibleSet(const ObjFuncl& obj, const Reg& reg) {
    return ComposedRegressorOps<Reg>::Project(reg);
  }

private:

  static std::vector<_ScaledShiftedSpike>
  FunclGradientFromFeatures(const learch_path& trainFeatures,
                            const learch_path& predFeatures) {
    std::vector<_ScaledShiftedSpike> result;
    for (int ii = 0; ii < trainFeatures.size(); ii++)
      result.push_back(_ScaledShiftedSpike(trainFeatures[ii], 1.0));
    for (int ii = 0; ii < predFeatures.size(); ii++)
      result.push_back(_ScaledShiftedSpike(predFeatures[ii], -1.0));
    return result;
  } 

};

#endif
