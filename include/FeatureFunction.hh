#ifndef __FEATURE_FUNCTION_H__
#define __FEATURE_FUNCTION_H__

#include "LEARCHCommon.hh"

/**
   \tparam Env The type of the environment; i.e., the context in which the 
              features are generated.
 */
template <class Env, class FeatureFunc>
class FeatureFunction {
public:
  
  /**
     \param featureFunc A reference to the feature function object
                        (which might contain parameters for feature 
                        generation, for example).
     \param environment A reference to the context in which the features 
                        are generated (e.g., a description of the world 
                        in which planning takes place).
     \param state A state
     \return A feature vector associated with the state
   */
  static learch_vector
  Eval(const FeatureFunc& featureFunc,
       const Env& environment, 
       const learch_vector& state);

};

#endif
