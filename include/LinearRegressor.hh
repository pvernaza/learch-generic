#ifndef __LINEAR_REGRESSOR_OPS_H__
#define __LINEAR_REGRESSOR_OPS_H__

#include "LEARCHCommon.hh"
#include "Regressor.hh"
#include <sstream>
#include <vector>

/**
   Parameters controlling the training of a linear regressor
 */
class LinearRegressorParams {
public:
  LinearRegressorParams(double _ridge) : ridge(_ridge) {};
  double ridge;
};

/**
   Definition of a linear regressor and operations on it
 */
class LinearRegressor {

  friend class BasicRegressorOps<LinearRegressor>;
  friend class DbgRegressorOps<LinearRegressor>;

public:

  LinearRegressor()
    : wtVec()
  { };

  explicit LinearRegressor(int nFeatures) 
    : wtVec(learch_vector::Zero(nFeatures))
  { }

  explicit LinearRegressor(const learch_vector& _wtVec) 
    : wtVec(_wtVec)
  { }

  learch_vector GetWeightVector() { return wtVec; }

private:

  const learch_vector wtVec;

};

template <>
class RegressorParamOps<LinearRegressorParams, LinearRegressor> {

public:

  static LinearRegressor
  Train(const LinearRegressorParams& regParams,
        const std::vector<std::pair<learch_vector, double> >& trainData) {

    assert(trainData.size() > 0);

    int nWeights = trainData[0].first.size();

    learch_vector ys(trainData.size());
    for (int i = 0; i < trainData.size(); i++)
      ys(i) = trainData[i].second;
    
    Eigen::MatrixXd aMat(trainData.size(), nWeights);
    for (int i = 0; i < trainData.size(); i++)
      aMat.row(i) = trainData[i].first;

    Eigen::MatrixXd lMat = aMat.transpose() * aMat 
      + regParams.ridge * Eigen::MatrixXd::Identity(nWeights, nWeights);

    learch_vector rhs = aMat.transpose() * ys;

    learch_vector wtVec = 
      lMat.fullPivLu().solve(rhs);
      //      rMat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(ys);
    
    return LinearRegressor(wtVec);
  }
};

template <>
class BasicRegressorOps<LinearRegressor> {
  
public:

  static double 
  Eval(const LinearRegressor& reg, const learch_vector& state) {
    if (reg.wtVec.size() == 0) return 0;
    return reg.wtVec.dot(state);
  }

};

template <> 
class DbgRegressorOps<LinearRegressor> {
public:

  static std::string
  ToString(LinearRegressor const& reg) {
    std::ostringstream result;
    result << reg.wtVec;
    return std::string("LinearRegressor( ") + result.str() + std::string(")");
  }

};

#endif
