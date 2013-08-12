#ifndef __OBJECTIVE_FUNCTIONAL_H__
#define __OBJECTIVE_FUNCTIONAL_H__

/**
   This class defines operations on an objective functional for
   functional gradient optimization.
   \tparam ObjFuncl The type of an objective functional
   \tparam Reg The type of a regressor (i.e., a function)
 */
template <typename ObjFuncl, typename Reg>
class ObjectiveFunctionalOps {

public:

  typedef std::pair<learch_vector, double> ScaledShiftedSpike;

  // FIXME: return object has potential to smash stack?
  static std::vector<ScaledShiftedSpike>
  Gradient(const ObjFuncl& obj, const Reg& reg);

  static Reg
  ProjectFeasibleSet(const ObjFuncl& obj, const Reg& reg);

};

#endif
