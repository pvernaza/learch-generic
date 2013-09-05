#ifndef __SUM_THRESH_REGRESSOR_H__
#define __SUM_THRESH_REGRESSOR_H__

#include <boost/lexical_cast.hpp>

/**
   A generic type of regressor that is closed under the operations of
   addition, scaling, and projection, given a base regressor.
   NB: BaseReg should be deep-copyable
 */
template <typename BaseReg>
class SumThreshRegressor {

public:

  // No pointers exposed or duplicated externally, and no references
  // to external pointers contained within. Therefore, each instance
  // of this class is responsible for deallocation of exactly those
  // pointers that it allocates and none others.

  // Represents the tree-structured regressor
  class RTree {
  public: 
    virtual double Eval(const learch_vector& state) const = 0;
    virtual RTree* Clone() const = 0;
    virtual ~RTree() {};
    virtual std::string AsString() const = 0;
  };

  // Sum of regressors
  class RSum : public RTree {
  public:
    RSum(const RTree& _n0, const RTree& _n1) 
      : n0(_n0.Clone()), n1(_n1.Clone()) {};
    ~RSum() { delete n0; delete n1; }
    RSum* Clone() const { return new RSum(*n0, *n1); }
    double Eval(const learch_vector& s) const { 
      return n0->Eval(s) + n1->Eval(s); 
    }
    std::string AsString() const { 
      using std::string;
      return "( " + n0->AsString() + " ) + ( " + n1->AsString() + " ) ";
    }
  private:
    RSum& operator=(RSum const& src) {};
    RSum(const RSum& n) : n0(n.n0.Clone()), n1(n.n1.Clone()) {};
    const RTree *n0, *n1;
  };
  
  // Scaled regressor
  class RScale : public RTree {
  public:
    RScale(const RTree& _n0, double _scale)
      : n0(_n0.Clone()), scale(_scale) {};
    ~RScale() { delete n0; }
    RScale* Clone() const { return new RScale(*n0, scale); }
    double Eval(const learch_vector& s) const { return scale * n0->Eval(s); }
    std::string AsString() const {
      return "( " + boost::lexical_cast<std::string>(scale) 
        + " * ( " + n0->AsString() + " ) )";
    }
  private:
    RScale& operator=(RScale const& src) {};
    RScale(const RScale& n) : n0(n.n0.Clone()), scale(n.scale){};
    const RTree* n0;
    const double scale;
  };

  // Thresholded regressor
  class RThresh : public RTree {
  public:
    RThresh(const RTree& _n0, double _threshold) 
      : n0(_n0.Clone()), threshold(_threshold) {};
    ~RThresh() { delete n0; }
    RThresh* Clone() const { return new RThresh(*n0, threshold); }
    double Eval(const learch_vector& s) const { 
      double raw = n0->Eval(s); 
      return raw > threshold ? raw : threshold; 
    }
    std::string AsString() const {
      return "THRESH( " + n0->AsString() + " ) ";
    }
  private:
    RThresh& operator=(RThresh const& src) {};
    RThresh(const RThresh& n) : n0(n.n0.Clone()), threshold(n.threshold) {};
    const RTree* n0;
    double threshold;
  };
  
  // Leaf node consisting of a base regressor
  class RReg : public RTree {
  public:
    explicit RReg(const BaseReg& _reg) : reg(_reg) {};
    ~RReg() { }
    RReg* Clone() const { return new RReg(reg); }
    double Eval(const learch_vector& s) const { 
      return BasicRegressorOps<BaseReg>::Eval(reg, s);
    };
    std::string AsString() const {
      return DbgRegressorOps<BaseReg>::ToString(reg);
    }
  private:
    RReg& operator=(RReg const& src) {};
    const BaseReg reg;
  };

  SumThreshRegressor(const BaseReg& reg, double _threshold)
    : regTree(new RReg(reg)), 
      threshold(_threshold) 
  {};

  SumThreshRegressor(const RTree& _regTree, double _threshold)
    : regTree(_regTree.Clone()),
      threshold(_threshold)
  {};

  SumThreshRegressor(const SumThreshRegressor& src)
    : regTree(src.regTree->Clone()),
      threshold(src.threshold)
  {};

  ~SumThreshRegressor() { 
    delete regTree;
  }

  SumThreshRegressor& operator=(const SumThreshRegressor& src) 
  {
    delete regTree;
    regTree = src.regTree->Clone();
    threshold = src.threshold;
    return *this;
  }

  typedef SumThreshRegressor Reg;

  double Eval(const learch_vector& s) const { 
    return regTree->Eval(s); 
  }

  Reg Scale(double scale) const { 
    RScale result(*regTree, scale);
    return Reg(result, threshold);
  }

  Reg Project() const {
    RThresh result(*regTree, threshold);
    return Reg(result, threshold); 
  }

  Reg Add(const Reg& r1) const { 
    RSum result(*regTree, *r1.regTree);
    return Reg(result, threshold);
  }

  string ToString() const {
    return regTree->AsString();
  }

private:

  const RTree* regTree;
  double threshold;

};

template <typename BaseReg>
class BasicRegressorOps< SumThreshRegressor<BaseReg> > {

public:

  typedef SumThreshRegressor<BaseReg> Reg;

  static double
  Eval(const Reg& r0, const learch_vector& state) { 
    return r0.Eval(state); 
  }
};

template <typename BaseReg>
class DbgRegressorOps< SumThreshRegressor<BaseReg> > {
  
public:

  static std::string
  ToString(SumThreshRegressor<BaseReg> const& reg) {
    return reg.ToString();
  }

};

/**
   Implementation of operations on a SumThreshRegressor
 */
template <typename BaseReg>
class ComposedRegressorOps< SumThreshRegressor<BaseReg> > : 
  public BasicRegressorOps< SumThreshRegressor<BaseReg> > {

public:

  typedef SumThreshRegressor<BaseReg> Reg;

  static Reg
  Add(const Reg& r0, const Reg& r1) {
    return r0.Add(r1);
  }

  static Reg
  Scale(const Reg& r0, double scale) {
    return r0.Scale(scale);
  }

  static Reg
  Project(const Reg& r0) {
    return r0.Project();
  }

};

/**
   Type of parameters corresponding to SumThreshRegressor
 */
template <class BaseRegParams>
struct SumThreshRegressorParams {
  SumThreshRegressorParams(const BaseRegParams& _baseRegParams,
                           double _threshold) 
    : baseRegParams(_baseRegParams),
      threshold(_threshold) {};
  BaseRegParams baseRegParams;
  double threshold;
};

/**
   Implementation of operations on a SumThreshRegressorParams
 */
template <typename BaseRegParams, typename BaseReg>
class RegressorParamOps< SumThreshRegressorParams<BaseRegParams>,
                         SumThreshRegressor<BaseReg> > {

public:

  static SumThreshRegressor<BaseReg>
  Train(const SumThreshRegressorParams<BaseRegParams>& params,
        const std::vector<std::pair<learch_vector, double> >& trainData) {
    SumThreshRegressor<BaseReg> result
      (RegressorParamOps<BaseRegParams,BaseReg>::Train
       (params.baseRegParams, trainData),
       params.threshold);
    return result;
  }

};

#endif
