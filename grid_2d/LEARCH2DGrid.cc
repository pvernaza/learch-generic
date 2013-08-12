/**

   LEARCH2DGrid.cc

   Description
   ==========

   This is example code showing how to apply the LEARCH library to
   learn costs for a simple 2D path planning problem. 

   Prerequisites
   ==========
   *) Boost (http://www.boost.org)
   *) Eigen (http://eigen.tuxfamily.org)
   *) The CImg library (http://cimg.sourceforge.net)
   *) ImageMagick (http://www.imagemagick.org/script/index.php)

   Compiling
   ==========
   1) Customize the include locations at the top of Makefile
   2) Type 'make'

   Usage
   ==========

   1) Run ./LEARCH2DGrid
   2) A window will pop up for each input image in sequence. Click and drag
      the mouse to define a training path for each input image.  Then close
      the window.
   3) The training will begin.  Every time the training algorithm calls 
      the path planner, a window will appear depicting the current 
      loss-augmented cost function for the current environment, along with
      the optimal path for this cost function.  Close the window to proceed.
   4) The training will stop after a fixed number of iterations.
   

   Detailed description
   ==========

   This is example code showing how to apply the LEARCH library to
   learn costs for a simple 2D path planning problem. As described in
   the documentation for LEARCHLearner, the library is applied to a
   new problem by providing definitions of several key types along
   with implementations of operations on these types.  The types and
   operations defined in this example are as follows.

   The first type is the type of the environment in which planning
   takes place, which is represented as the type parameter Env in
   LEARCHLearner.  In this example, the type of the environment is the
   Grid2DEnvironment class, which stores a 2D image representing a map
   from which features are later derived (along with some precomputed
   feature maps).  To give another example, in the case of arm
   planning, the environment might consist of a class containing a
   description of the environment in the form of a point cloud and/or
   the locations and descriptions of objects.

   The type of the feature function is given by the class
   Grid2DBlurFeatures.  Since an object of this type is passed to the
   feature evaluation function, it may be used to store various
   parameters used to compute the features.  Since we don't need to
   pass such parameters in this case, the class is empty.

   Next, we provide the implementation of the feature function applied
   to our specific environment type (Grid2DEnvironment) and our
   specific feature type (Grid2DBlurFeatures) by providing an
   implementation (template specialization) of the class
   FeatureFunction<Grid2DEnvironment, Grid2DBlurFeatures>.  The method
   Eval of this class is applied to an instance of the environment and
   a specific state (and a feature function object), and it returns a
   feature vector.  The feature vector in our case consists of the
   values of several Gaussian blur filters run over the image
   (environment), evaluated at the given state.

   Finally, a specialization of LEARCHPlanner to the Grid2DEnvironment
   type is provided in order to implement the planner.  In our case,
   this works by evaluating the input cost function at every point in
   the input environment, creating a grid of costs with the same
   dimensions as the input environment.  This grid is then passed to
   an external implementation of the Fast Marching Method, which
   computes the optimal path with respect to this cost function.

   LEARCHLearner is parameterized over the type of the environment
   (Grid2DEnvironment), the type of the regressor used to update the
   cost function (SumThreshRegressor<LinearRegressor>), the type of
   the feature function (Grid2DBlurFeatures), and the type of the loss
   function (ExpMinDistToPathLoss).  The regressor and loss function
   used here are implemented as part of the library and should suffice
   for future applications; however, these can be replaced if
   necessary.

 */

#include "FeatureFunction.hh"
#include "FunctionalOptimizer.hh"
#include "fastMarching.hh"
#include "LEARCHPlanner.hh"
#include "LEARCHObjective.hh"
#include "LinearRegressor.hh"
#include "StepRegressor.hh"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include "CImg.h"

#define TEST_IMAGE_FILES "grid_2d/testMap0.jpg grid_2d/testMap1.jpg grid_2d/testMap2.jpg"
#define N_BLUR_FEATURES 10
#define FEATURE_MIN_BLUR 1.0
#define FEATURE_MAX_BLUR 40.0
#define LOSS_FUNC_SIGMA 20.0
#define LEARCH_LEARN_RATE 10.0
#define LEARCH_ITERATIONS 100
#define GRID_CELL_SIZE 1
#define FMM_MIN_COST 1e-6
#define LEARCH_MIN_COST 1.0

#define DISPLAY_SIZE_X 600
#define DISPLAY_SIZE_Y 600

using namespace cimg_library;

/**
   The concrete type of the environment.
   Contains a map as well as blurred versions of it for 
   feature extraction.
*/
class Grid2DEnvironment {

public:

  Grid2DEnvironment(const CImg<double>& _mapImage,
                    int nBlurLevels, double minBlur, double maxBlur)
    : mapImage(_mapImage) {
    for (int ii = 0; ii < nBlurLevels; ii++) {
      double sigma = minBlur + ii * (maxBlur - minBlur) / (nBlurLevels-1);
      blurredMaps.push_back(mapImage.get_blur(sigma, sigma, true, true));
    }
  }

  CImg<double> mapImage;
  std::vector<CImg<double> > blurredMaps;
};

/**
   The concrete type of the feature function
*/
class Grid2DBlurFeatures {
  
};

/**
   The implementation of the feature function operations
 */
template<>
class FeatureFunction<Grid2DEnvironment, Grid2DBlurFeatures> {

public:

  static learch_vector
  Eval(const Grid2DBlurFeatures& featureFunc,
       const Grid2DEnvironment& env, 
       const learch_vector& state) {
    const int nFeatures = env.blurredMaps.size() * 3 + 1;
    learch_vector result(nFeatures);
    for (int ii = 0; ii < env.blurredMaps.size(); ii++)
      for (int ci = 0; ci < 3; ci++) 
        result[ii*3+ci] = (env.blurredMaps[ii])(state(0), state(1), 0, ci);
    result[nFeatures-1] = 1;    // bias feature
    return result;
  }

};

/**
   The implementation of the planner using the Fast Marching Method.
 */
template<>
class LEARCHPlanner<Grid2DEnvironment> {

public:

  static learch_path*
  Plan(const Grid2DEnvironment& env,
       const LEARCHPlannerCost<Grid2DEnvironment>& costFunc,
       const learch_vector& start,
       const learch_vector& goal) {
    printf("Evaluating loss-augmented cost function\n");
    vector<int> gridSize(2);
    gridSize[0] = env.mapImage.width();
    gridSize[1] = env.mapImage.height();
    double costGridMem[gridSize[0]][gridSize[1]];
    for (int x = 0; x < gridSize[0]; x++) {
      for (int y = 0; y < gridSize[1]; y++) {
        learch_vector state(2);
        state << x, y;
        costGridMem[x][y] = costFunc.Eval(env, state) + FMM_MIN_COST;
      }
    }
    printf("Calling fast marching method\n");
    vector<double> fmmCostDx(2);   // dimensions of a grid cell
    fmmCostDx[0] = GRID_CELL_SIZE;
    fmmCostDx[1] = GRID_CELL_SIZE;
    vector<int> fmmGoal(2);        // goal location
    fmmGoal[0] = goal(0);
    fmmGoal[1] = goal(1);
    vector<int> fmmStop;
    double valfMem[gridSize[0]][gridSize[1]];
    ArrayND<double> fmmCostGrid(&costGridMem[0][0], gridSize, ARRAY_C_LAYOUT);
    ArrayND<double> fmmValf(&valfMem[0][0], gridSize, ARRAY_C_LAYOUT);
    fastMarching(fmmCostGrid, fmmCostDx, fmmGoal, fmmStop, fmmValf);
    learch_path path0(1, start);
    learch_path result = FindFMMPath(fmmValf, path0);
    DebugViewPlan(env, fmmCostGrid, result);
    return (new learch_path(result));
  }

private:
  
  // find the path given the FMM value function 
  static learch_path
  FindFMMPath(const ArrayND<double>& fmmValf, const learch_path path0) {
    vector<int> dims = fmmValf.getDims();
    learch_vector curPos = path0[path0.size()-1];
    std::vector<int> curPosInt(2), nextPosInt(2);
    curPosInt[0] = curPos(0);
    curPosInt[1] = curPos(1);
    double minVal = INFINITY;
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        std::vector<int> nbPosInt(2);
        nbPosInt[0] = curPosInt[0] + dx;
        nbPosInt[1] = curPosInt[1] + dy;
        if (nbPosInt[0] >= 0 && nbPosInt[0] < dims[0] &&
            nbPosInt[1] >= 0 && nbPosInt[1] < dims[1] &&
            fmmValf.get(nbPosInt) < minVal) {
          nextPosInt = nbPosInt;
          minVal = fmmValf.get(nbPosInt);
        }
      }
    }
    // stop if we cannot descend the value function
    if (minVal == fmmValf.get(curPosInt)) return path0;
    learch_vector newPos(2);
    newPos << nextPosInt[0], nextPosInt[1];
    learch_path newPath(path0);
    newPath.push_back(newPos);
    return FindFMMPath(fmmValf, newPath);
  }

  // debugging functions

  static CImg<double>
  ArrayNDToCImg(const ArrayND<double>& cost) {
    vector<int> dims = cost.getDims();
    CImg<double> result(dims[0], dims[1], 1, 3);
    for (int x = 0; x < dims[0]; x++) {
      for (int y = 0; y < dims[1]; y++) {
        vector<int> loc(2);
        loc[0] = x; loc[1] = y;
        double viewCost = log(cost.get(loc) + 1e-12);
        result(x,y,0,0) = viewCost;
        result(x,y,0,1) = viewCost;
        result(x,y,0,2) = viewCost;
      }
    }
    return result;
  }

  static void
  SuperimposePath(const learch_path& path,
                  CImg<double>& img) {                  
    double imMax = img.max();
    for (int ii = 0; ii < path.size(); ii++) {
      img((path[ii])(0), (path[ii])(1), 0, 0) = imMax;
      img((path[ii])(0), (path[ii])(1), 0, 1) = 0.;
      img((path[ii])(0), (path[ii])(1), 0, 2) = 0.;
    }
    img.resize(DISPLAY_SIZE_X, DISPLAY_SIZE_Y);
  }

  static void
  DebugViewPlan(const Grid2DEnvironment& env,
                const ArrayND<double>& costGrid,
                const learch_path& path) {
    CImg<double> costImg(ArrayNDToCImg(costGrid));
    CImg<double> mapImg(env.mapImage);
    SuperimposePath(path, costImg);
    SuperimposePath(path, mapImg);
    CImgDisplay displayCost(costImg, "Optimal loss-augmented-cost path", 1);
    CImgDisplay displayMap(mapImg, "Optimal loss-augmented-cost path", 1);
    while (!displayCost.is_closed()) {
      displayCost.wait();
      if (displayCost.button()) {
        printf("map value = %g\n", costImg(displayCost.mouse_x(), 
                                           displayCost.mouse_y()));
      }
    }
  }

};

/**
   Methods to retrieve training data from input images
 */
class TrainingDataRetriever {

public:

  typedef std::pair<Grid2DEnvironment, learch_path> EnvAndPath;

  static std::vector<std::string>
  ParseTrainingFilenames(const std::string& concatNames) {
    std::vector<std::string> result;
    boost::split(result, concatNames, boost::is_any_of(" "));
    return result;
  }

  static Eigen::MatrixXd 
  CImgToEigenMatrix(const CImg<double>& image) {
    Eigen::MatrixXd result(image.height(), image.width());
    for (int x = 0; x < image.width(); x++)
      for (int y = 0; y < image.height(); y++) 
        result(x,y) = image(x,y,0,0);
    return result;
  }

  static learch_path
  GetPathFromClicks(CImgDisplay& display) {
    learch_path path;
    while (!display.is_closed()) {
      display.wait();
      if (display.button()) {
        learch_vector xyLoc(2);
        xyLoc << display.mouse_x(), display.mouse_y();
        path.push_back(xyLoc);
        printf("click at %g, %g\n", xyLoc(0), xyLoc(1));
      }
    }
    return path;
  }

  static std::vector<EnvAndPath>*
  GetTrainingData(const std::vector<std::string>& fnames) {
    std::vector<EnvAndPath>* result = new std::vector<EnvAndPath>();
    for (int iImage = 0; iImage < fnames.size(); iImage++) {
      CImg<double> image(fnames[iImage].c_str());
      image += 1.0;
      CImgDisplay display(image, "Click and drag to define training path");
      cout << "Click and drag to define training path\n";
      Grid2DEnvironment 
        env(image, N_BLUR_FEATURES, FEATURE_MIN_BLUR, FEATURE_MAX_BLUR);
      EnvAndPath envAndPath(env, GetPathFromClicks(display));
      result->push_back(envAndPath);
    }
    return result;
  }

};

class MainProgram {

public:

   // Type aliases
  typedef Grid2DEnvironment Env;
  //  typedef LinearRegressor BaseReg;
  //  typedef LinearRegressorParams BaseRegParams;
  typedef StepRegressor<LinearRegressor> BaseReg;
  typedef StepRegressorParams<LinearRegressorParams> BaseRegParams;
  typedef SumThreshRegressor<BaseReg> Reg;
  typedef SumThreshRegressorParams<BaseRegParams> RegParams;
  typedef Grid2DBlurFeatures FeatureFunc;
  typedef ExpMinDistToPathLoss LossFunc;

  // The specific objective functional
  typedef LEARCHObjective<Env, Reg, FeatureFunc, LossFunc> ObjFuncl;
  // The specific functional optimizer
  //  typedef NaiveFunctionalOptimizer FunclOpt;
  typedef RepeatedGradientFunctionalOptimizer FunclOpt;
  // Operations on the functional optimizer
  typedef FunctionalOptimizerOps<FunclOpt, ObjFuncl, RegParams, Reg> FunclOptOps;

  static void Run() {
    // Get the training data
    std::vector<std::string> filenames = 
      TrainingDataRetriever::ParseTrainingFilenames(TEST_IMAGE_FILES);
    std::vector<TrainingDataRetriever::EnvAndPath>* rawTrainData =
      TrainingDataRetriever::GetTrainingData(filenames);
    
    // Initialize feature/loss/cost functions, objective functional,
    // and functional optimizer
    FeatureFunc featureFunc;
    LossFunc lossFunc(LOSS_FUNC_SIGMA);
    ObjFuncl objFuncl(*rawTrainData, featureFunc, lossFunc);
    FunclOpt funclOpt(LEARCH_LEARN_RATE);
    //    BaseReg baseReg;
    //    BaseRegParams baseRegParams;
    BaseReg baseReg((LinearRegressor()));
    BaseRegParams baseRegParams((LinearRegressorParams()));
    Reg costFunc0(baseReg, LEARCH_MIN_COST);
    RegParams regParams(baseRegParams, LEARCH_MIN_COST);






    // TESTING
    // TESTING
    // TESTING
    learch_vector test;
    BasicRegressorOps<StepRegressor<LinearRegressor> >::Eval(baseReg, test);
    BasicRegressorOps<Reg>::Eval(costFunc0, test);




    FunclOptOps::Optimize(funclOpt, objFuncl, regParams, costFunc0, LEARCH_ITERATIONS);
  }

};

int main() {
  MainProgram::Run();
}
