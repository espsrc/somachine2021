// /opt/spark/bin/spark-shell --packages JMailloH:kNN_IS:3.0,djgg:PCARD:1.3

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

// Check Train & Test

train.persist
train.count	//Triggers all the previous transformations + cache
train.first

test.persist
test.count
test.first

// Class balance
val classInfo = train.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()

// Statistics

import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

val summaryTrain: MultivariateStatisticalSummary = Statistics.colStats(train.map(_.features))

var outputString = new ListBuffer[String]
outputString += "******TRAIN******\n\n"
outputString += "@Max (0) --> " + summaryTrain.max(0) + "\n"
outputString += "@Min (0) --> " + summaryTrain.min(0) + "\n"
outputString += "@Mean (0) --> " + summaryTrain.mean(0) + "\n"
outputString += "@Variance (0) --> " + summaryTrain.variance(0) + "\n"
outputString += "@NumNonZeros (0) --> " + summaryTrain.numNonzeros(0) + "\n"

// Correlation

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics

// calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method
// If a method is not specified, Pearson's method will be used by default.
val correlMatrix: Matrix = Statistics.corr(train.map(_.features), "pearson")
println(correlMatrix.toString)


// DT Benchmark

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32

val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)


// DT Prediction

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()

println(s"Test Accuracy = $testAcc")



// Random Forest Benchmark

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel

// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 100
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainClassifier(train, numClasses,categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,maxDepth, maxBins)

// RF Prediction

// Evaluate model on test instances and compute test error
val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val testAcc = 1 - labelAndPreds.filter(r => r._1 !=r._2).count.toDouble / test.count()

println(s"Test Accuracy = $testAcc")


// kNN Benchmark

import org.apache.spark.mllib.classification.kNN_IS.kNN_IS

val k = 3

val numClass = train.map(_.label).distinct().collect().length
val numFeatures = train.first().features.size

val knn = kNN_IS.setup(train, test, k, 2, numClass, numFeatures, train.getNumPartitions, 2, -1, 1)

val predictions = knn.predict(sc)

// Metrics

import org.apache.spark.mllib.evaluation._

val metrics = new MulticlassMetrics(predictions)
val precision = metrics.precision
val cm = metrics.confusionMatrix

val binaryMetrics = new BinaryClassificationMetrics(predictions)
val AUC = binaryMetrics.areaUnderROC


// PCARD Benchmark

import org.apache.spark.mllib.tree.PCARD

val cuts = 5 // bins for discretization
val trees = 10 // iterations

val pcardTrain = PCARD.train(train, trees, cuts)
val pcard = pcardTrain.predict(test)

// PCARD Prediction

val labels = test.map(_.label).collect()

var cont = 0

for (i <- labels.indices) {
  if (labels(i) == pcard(i)) {
    cont += 1
  }
}

val testAcc = cont / labels.length.toFloat

println(s"Test Accuracy = $testAcc")
