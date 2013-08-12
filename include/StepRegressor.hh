#ifndef __STEP_REGRESSOR_HH__
#define __STEP_REGRESSOR_HH__

#include <algorithm>
#include <functional>
#include <boost/bind.hpp>

template <typename BaseRegParams>
class StepRegressorParams {
public:
  StepRegressorParams(const BaseRegParams& _baseRegParams) 
    : baseRegParams(_baseRegParams) {};
  const BaseRegParams baseRegParams;
};

/**
   A regressor that is constant on all the sub- and super-level sets
   of a certain level set of a given base regressor.  The precise
   level set and constants are chosen in such a way as to minimize the
   variance of the error with respect to the base regressor's
   prediction.

   Assumes that BaseReg is safe to copy.
 */
template <typename BaseReg>
class StepRegressor {

public: 
  friend class BasicRegressorOps<StepRegressor<BaseReg> >;

  StepRegressor(const BaseReg& _baseReg) 
    : baseReg(_baseReg), levelThresh(0), 
      belowConst(0), aboveConst(0) {};

  StepRegressor(const BaseReg& _baseReg, 
                double _levelThresh,
                double _belowConst, double _aboveConst)
    : baseReg(_baseReg), 
      levelThresh(_levelThresh),
      belowConst(_belowConst), 
      aboveConst(_aboveConst) {};

private:
  const BaseReg baseReg;
  double levelThresh;         // level set that is prediction boundary
  double belowConst;           // prediction on lower side of boundary
  double aboveConst;           // prediction on upper side of boundary
};

template <typename BaseReg>
class BasicRegressorOps<StepRegressor<BaseReg> > {

public:
  static double
  Eval(const StepRegressor<BaseReg>& reg,
       const learch_vector& state) {
    double baseRegOutput = BasicRegressorOps<BaseReg>::Eval(reg.baseReg, state);
    if (baseRegOutput < reg.levelThresh) return reg.belowConst;
    else return reg.aboveConst;
  }

};

template <typename BaseRegParams, typename BaseReg>
class RegressorParamOps<StepRegressorParams<BaseRegParams>, 
                        StepRegressor<BaseReg> > {
  
public:

  typedef std::pair<learch_vector, double> TrainDatum;
  typedef std::vector<TrainDatum> TrainData;

  // FIXME: remove redundant classifier evaluations by caching
  static StepRegressor<BaseReg>
  Train(const StepRegressorParams<BaseRegParams>& regParams,
        const TrainData& trainData) {
    BaseReg baseReg = 
      RegressorParamOps<BaseRegParams,BaseReg>
      ::Train(regParams.baseRegParams, trainData);
    double threshold;
    double cLo, cHi;
    FindOptimalThreshold(trainData, baseReg, threshold, cLo, cHi);
    return StepRegressor<BaseReg>(baseReg, threshold, cLo, cHi);
  }

private:

  static bool
  BaseRegComparator(const BaseReg& baseReg,
                    const TrainDatum& d0, 
                    const TrainDatum& d1) {
    return BasicRegressorOps<BaseReg>::Eval(baseReg, d0.first) 
      < BasicRegressorOps<BaseReg>::Eval(baseReg, d1.first);
  }

  static void 
  FindOptimalThreshold(const TrainData& trainData,
                       const BaseReg& baseReg,
                       double& threshold,
                       double& cLo,
                       double& cHi) {

    assert(trainData.size() > 0);
    TrainData* const sortedTrainData = new TrainData(trainData);
    std::sort(sortedTrainData->begin(), sortedTrainData->end(), 
              boost::bind(BaseRegComparator, baseReg, _1, _2));

    std::vector<double> sortedOutputs(trainData.size());
    for (int i = 0; i < sortedOutputs.size(); i++) 
      sortedOutputs[i] = BasicRegressorOps<BaseReg>
        ::Eval(baseReg, (*sortedTrainData)[i].first);

    // ith element = variance of all outputs \leq to this output
    std::vector<double> varBelow(trainData.size()); 
    // ith element = variance of all outputs \geq to this output
    std::vector<double> varAbove(trainData.size());
    std::vector<double> meanBelow(trainData.size());
    std::vector<double> meanAbove(trainData.size());
    varBelow[0] = 0;
    varAbove[varAbove.size()-1] = 0;
    meanBelow[0] = sortedOutputs[0];
    meanAbove[meanAbove.size()-1] = 0;

    for (int ii = 1; ii < varBelow.size() - 1; ii++) {
      double output = sortedOutputs[ii];
      meanBelow[ii] = (ii * meanBelow[ii-1] + output) / (ii + 1);
      double delta = output - meanBelow[ii];
      varBelow[ii] = (ii * varBelow[ii-1] + delta * delta) / (ii + 1);
    }

    for (int ii = varAbove.size() - 2; ii >= 0; ii--) {
      double output = sortedOutputs[ii];
      int nAbove = varAbove.size() - ii;
      meanAbove[ii] = (nAbove * meanAbove[ii+1] + output) / (nAbove + 1);
      double delta = output - meanAbove[ii];
      varAbove[ii] = (nAbove * varAbove[ii+1] + delta * delta) / (nAbove + 1);
    }

    double minVar = INFINITY;
    double minInd = -1;
    // ith iteration has boundary = output of (i+1)th element
    for (int ii = 0; ii < trainData.size() - 2; ii++) {
      double totVar = varBelow[ii] + varAbove[ii+1];
      if (totVar < minVar) {
        minVar = totVar;
        minInd = ii;
      }
    }

    // outputs \geq threshold get prediction cHi, rest get prediction cLo
    threshold = sortedOutputs[minInd];
    cLo = meanBelow[minInd];
    cHi = meanAbove[minInd+1];

    delete sortedTrainData;

    return;
  }

};

#endif
