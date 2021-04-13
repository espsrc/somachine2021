// /opt/spark/bin/spark-shell --packages JMailloH:kNN_IS:3.0,djgarcia:SmartReduction:1.0,djgarcia:Equal-Width-Discretizer:1.0 --jars /home/administrador/datasets/mdlp-mrmr.jar

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}

sc.setLogLevel("ERROR")

// Load Train & Test

val pathTrain = "file:///home/administrador/datasets/susy-10k-tra.data"
val rawDataTrain = sc.textFile(pathTrain)

val pathTest = "file:///home/administrador/datasets/susy-10k-tst.data"
val rawDataTest = sc.textFile(pathTest)

// Train & Test RDDs

val train = rawDataTrain.map{line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}.repartition(16)

val test = rawDataTest.map { line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}.repartition(16)

train.persist
test.persist


// Encapsulate Learning Algorithms

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

def trainDT(train: RDD[LabeledPoint], test: RDD[LabeledPoint], maxDepth: Int = 5): Double = {
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxBins = 32

    val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    val labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    testAcc
}


import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.mllib.evaluation._
import org.apache.spark.rdd.RDD


def trainKNN(train: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int = 3): Double = {

    val numClass = train.map(_.label).distinct().collect().length
    val numFeatures = train.first().features.size

    val knn = kNN_IS.setup(train, test, k, 2, numClass, numFeatures, train.getNumPartitions, 2, -1, 1)
    val predictions = knn.predict(sc)
    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.precision

    precision

}


/*****Instance Selection*****/

// FCNN_MR

import org.apache.spark.mllib.feature._

val k = 3 //number of neighbors

val fcnn_mr_model = new FCNN_MR(train, k)

val fcnn_mr = fcnn_mr_model.runPR()

fcnn_mr.persist()

fcnn_mr.count()


trainDT(fcnn_mr, test)

trainKNN(fcnn_mr, test)


// RMHC_MR

import org.apache.spark.mllib.feature._

val p = 0.1 // Percentage of instances (max 1.0)
val it = 100 // Number of iterations
val k = 3 // Number of neighbors

val rmhc_mr_model = new RMHC_MR(train, p, it, k, 48151623)

val rmhc_mr = rmhc_mr_model.runPR()

rmhc_mr.persist()

rmhc_mr.count()

trainDT(rmhc_mr, test)

trainKNN(rmhc_mr, test)


// SSMA-SFLSDE_MR

import org.apache.spark.mllib.feature._

val ssmasflsde_mr_model = new SSMASFLSDE_MR(train) 

val ssmasflsde_mr = ssmasflsde_mr_model.runPR()

ssmasflsde_mr.persist()

ssmasflsde_mr.count()

trainDT(ssmasflsde_mr, test)

trainKNN(ssmasflsde_mr, test)



/*****Discretization*****/


// Equal Width Discretizer

import org.apache.spark.mllib.feature._

val nBins = 25 // Number of bins

val discretizerModel = new EqualWidthDiscretizer(train,nBins).calcThresholds()

val discretizedTrain = discretizerModel.discretize(train)
val discretizedTest = discretizerModel.discretize(test)

discretizedTrain.first
discretizedTest.first

trainDT(discretizedTrain, discretizedTest)

trainKNN(discretizedTrain, discretizedTest)



// MDLP

import org.apache.spark.ml.feature.{MDLPDiscretizer, LabeledPoint => NewLabeledPoint}

val mdlpTrain = train.map(l => NewLabeledPoint(l.label, l.features.asML)).toDS()
val mdlpTest = test.map(l => NewLabeledPoint(l.label, l.features.asML)).toDS()

val bins = 25

val discretizer = new MDLPDiscretizer().setMaxBins(bins).setMaxByPart(10000).setInputCol("features").setLabelCol("label").setOutputCol("buckedFeatures")

val model = discretizer.fit(mdlpTrain)

val trainDisc = model.transform(mdlpTrain).rdd.map(row => LabeledPoint(
  row.getAs[Double]("label"),
  Vectors.dense(row.getAs[org.apache.spark.ml.linalg.Vector]("buckedFeatures").toArray)
))
val testDisc = model.transform(mdlpTest).rdd.map(row => LabeledPoint(
  row.getAs[Double]("label"),
  Vectors.dense(row.getAs[org.apache.spark.ml.linalg.Vector]("buckedFeatures").toArray)
))


trainDT(trainDisc, testDisc)

trainKNN(trainDisc, testDisc)


/*****Feature Selection*****/

//ChiSq

import org.apache.spark.mllib.feature.ChiSqSelector

val numFeatures = 5
val selector = new ChiSqSelector(numFeatures)
val transformer = selector.fit(train)

val chisqTrain = train.map { lp => 
  LabeledPoint(lp.label, transformer.transform(lp.features)) 
}

val chisqTest = test.map { lp => 
  LabeledPoint(lp.label, transformer.transform(lp.features)) 
}

chisqTrain.first.features.size

trainDT(chisqTrain, chisqTest)

trainKNN(chisqTrain, chisqTest)


// PCA

import org.apache.spark.mllib.feature.PCA

val numFeatures = 5

val pca = new PCA(5).fit(train.map(_.features))

val projectedTrain = train.map(p => p.copy(features = pca.transform(p.features)))
val projectedTest = test.map(p => p.copy(features = pca.transform(p.features)))

projectedTrain.first.features.size
projectedTest.first.features.size


trainDT(projectedTrain, projectedTest)

trainKNN(projectedTrain, projectedTest)


// mRMR

import org.apache.spark.mllib.feature._

val criterion = new InfoThCriterionFactory("mrmr")
val nToSelect = 5
val nPartitions = trainDisc.getNumPartitions

val featureSelector = new InfoThSelector(criterion, nToSelect, nPartitions).fit(trainDisc)

val reducedTrain = trainDisc.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
reducedTrain.first()

val reducedTest = testDisc.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))


trainDT(reducedTrain, reducedTest)

trainKNN(reducedTrain, reducedTest)
