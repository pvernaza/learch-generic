#ifndef __REGRESSOR_H__
#define __REGRESSOR_H__

#include "LEARCHCommon.hh"

#include <vector>

/**
   Class defining operations on regressor parameters
 */
template <typename RegParams, typename Reg>
class RegressorParamOps {
  
public:

  /**
     \param regParams Parameters to pass to the training procedure
     \param trainData vector of (vector input, scalar output) pairs
   */
  static Reg
  Train(const RegParams& regParams, 
        const std::vector<std::pair<learch_vector, double> >& trainData);

};

/**
   Class defining operations on a regressor.
   \tparam Reg The type of regressor to which these operations apply
 */
template <typename Reg>
class BasicRegressorOps {
  
public:

  static double
  Eval(const Reg& reg, const learch_vector& state);

};

template <typename Reg>
class DbgRegressorOps {

public:

  static std::string
  ToString(Reg const& reg);

};

/**
   A type of regressor that is closed under misc. operations
 */
template <typename Reg>
class ComposedRegressorOps : BasicRegressorOps<Reg> {

public:

  static Reg
  Add(const Reg& r0, const Reg& r1);

  static Reg
  Scale(const Reg& reg, double scale);
  
  static Reg
  Project(const Reg& reg);

};

#endif
