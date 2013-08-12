#ifndef __LEARCH_PLANNER_H__
#define __LEARCH_PLANNER_H__

/**
   The cost function passed to the planner
   \tparam Env The type of the planning environment
 */
template <class Env>
class LEARCHPlannerCost {
  
public:

  virtual double Eval(const Env& environment, 
                      const learch_vector& state) const = 0;

};

/**
   Interface for the planner to be called by the LEARCH training procedure. 
   Implemented by template specialization to each concrete type of Env.
 */
template <class Env>
class LEARCHPlanner {
  
public:

  /**
     \param environment The environment (context) in which to plan
     \param costFunc The cost function
     \param start The start state
     \param goal The goal state
     \return A pointer to a path. Caller responsible for deallocation.
   */
  static learch_path*
  Plan(const Env& environment,
       const LEARCHPlannerCost<Env>& costFunc,
       const learch_vector& start,
       const learch_vector& goal);

};

#endif
