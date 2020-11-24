// /opt/spark/bin/spark-shell --packages djgarcia:NoiseFramework:1.2,djgarcia:RandomNoise:1.0,djgarcia:SmartFiltering:1.0,JMailloH:Smart_Imputation:1.0,JMailloH:kNN_IS:3.0

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


// Min & Max values

import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

val fullDataset = train.union(test)

val summary = Statistics.colStats(fullDataset.map(_.features))

summary.min

summary.max


// Normalize Train & Test

val normalizedTrain = train.map{l =>
  val featuresArray = l.features.toArray.zipWithIndex.map{case (v,k) =>
    (v - summary.min(k)) / (summary.max(k) - summary.min(k))
  }
  LabeledPoint(l.label, Vectors.dense(featuresArray))
}

val normalizedTest = test.map{l =>
  val featuresArray = l.features.toArray.zipWithIndex.map{case (v,k) =>
    (v - summary.min(k)) / (summary.max(k) - summary.min(k))
  }
  LabeledPoint(l.label, Vectors.dense(featuresArray))
}


// Check Train & Test

val summaryTrain = Statistics.colStats(normalizedTrain.map(_.features))
summaryTrain.min
summaryTrain.max

val summaryTest = Statistics.colStats(normalizedTest.map(_.features))
summaryTest.min
summaryTest.max

val summaryUnion = Statistics.colStats(normalizedTrain.union(normalizedTest).map(_.features))
summaryUnion.min
summaryUnion.max

// DT & kNN Results

trainDT(normalizedTrain, normalizedTest)

trainKNN(normalizedTrain, normalizedTest)


// Add MVs

val mv_pct = 30 // 30% of MVs

val tam = rawDataTrain.count.toInt // Number of instances

val num = math.round(tam * (mv_pct.toDouble / 100)) // Number of MVs

val range = util.Random.shuffle(0 to tam - 1) // Random number gen.

val indices = range.take(num.toInt) // Random instances

val broadcastInd = rawDataTrain.sparkContext.broadcast(indices)

import scala.util.Random

val mvData = rawDataTrain.zipWithIndex.map {
  case (v, k) =>
    if (broadcastInd.value contains (k)) {
      val features = v.split(",").init
      val label = v.split(",").last
      val mv = features.indexOf(Random.shuffle(features.toList).head)
      features(mv) = "?"
      features.mkString(",").concat("," + label)
    } else {
      v
    }
}
mvData.persist

val mv_num = mvData.filter(_.contains("?")).count


// Remove MVs

val train_without_mv = mvData.filter(!_.contains("?"))

val trainMV = train_without_mv.map{line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}

// Train DT & kNN

trainMV.persist
trainMV.count

trainDT(trainMV, test)

trainKNN(trainMV, test)


// Mean Imputation

val numFeatures = train.first().features.size
var means: Array[Double] = new Array(numFeatures)

for(x <- 0 to numFeatures-1){
  means(x) = mvData.map(_.split(",")(x)).filter(v => !v.contains("?")).map(_.toDouble).mean
}

val meanImputedData = mvData.map(_.split(",").zipWithIndex.map{case (v,k)=> if (v == "?") means(k) else v.toDouble})

val mv_num = meanImputedData.filter(_.contains("?")).count

val trainMean = meanImputedData.map{arrayDouble =>
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}


// Train DT & kNN

trainMean.persist
trainMean.count

trainDT(trainMean, test)

trainKNN(trainMean, test)


// kNNI

import org.apache.spark.mllib.preprocessing.kNNI_IS.KNNI_IS

val k = 3
val pathHeader = "/home/administrador/datasets/susy.header"

val knni = KNNI_IS.setup(mvData, k, 2, pathHeader, mvData.getNumPartitions, "local")
val imputedData = knni.imputation(sc)

val mv_num_knni = imputedData.filter(_.contains("?")).count

val trainKNNI = imputedData.map{array =>
  val arrayDouble = array.map(f => f.toDouble) 
  val featureVector = Vectors.dense(arrayDouble.init) 
  val label = arrayDouble.last 
  LabeledPoint(label, featureVector)
}

// Train DT & kNN

trainKNNI.persist
trainKNNI.count


trainDT(trainKNNI, test)

trainKNN(trainKNNI, test)




/*****Noise Filtering*****/

import org.apache.spark.mllib.util._

val noise = 20 //(in %)

val noisyModel = new RandomNoise(train, noise)

val noisyData = noisyModel.runNoise()

noisyData.persist()

noisyData.count()


trainDT(train, test, 20)

trainKNN(train, test)


trainDT(noisyData, test, 20)

trainKNN(noisyData, test)


// ENN_BD

import org.apache.spark.mllib.feature._

val k = 3 //number of neighbors

val enn_bd_model = new ENN_BD(noisyData, k)

val enn_bd = enn_bd_model.runFilter()

enn_bd.persist()

enn_bd.count()

trainDT(enn_bd, test, 20)

trainKNN(enn_bd, test)


// NCNEdit_BD

import org.apache.spark.mllib.feature._

val k = 3 //number of neighbors

val ncnedit_bd_model = new NCNEdit_BD(noisyData, k)

val ncnedit_bd = ncnedit_bd_model.runFilter()

ncnedit_bd.persist()

ncnedit_bd.count()


trainDT(ncnedit_bd, test, 20)

trainKNN(ncnedit_bd, test)


// RNG_BD

import org.apache.spark.mllib.feature._

val order = true // Order of the graph (true = first, false = second)
val selType = true // Selection type (true = edition, false = condensation)

val rng_bd_model = new RNG_BD(noisyData, order, selType)

val rng_bd = rng_bd_model.runFilter()

rng_bd.persist()

rng_bd.count()

trainDT(rng_bd, test, 20)

trainKNN(rng_bd, test)


// HME_BD

import org.apache.spark.mllib.feature._

val nTrees = 100
val maxDepthRF = 10
val partitions = 4

val hme_bd_model = new HME_BD(noisyData, nTrees, partitions, maxDepthRF, 48151623)

val hme_bd = hme_bd_model.runFilter()

hme_bd.persist()

hme_bd.count()

trainDT(hme_bd, test, 20)

trainKNN(hme_bd, test)


// HME_BD Clean Data

val hme_bd_model_clean = new HME_BD(train, nTrees, partitions, maxDepthRF, 48151623)

val hme_bd_clean = hme_bd_model_clean .runFilter()

hme_bd_clean.persist()

hme_bd_clean.count()

trainDT(hme_bd_clean, test, 20)

trainKNN(hme_bd_clean, test)


// HTE_BD

import org.apache.spark.mllib.feature._

val nTrees = 100
val maxDepthRF = 10
val partitions = 4
val vote = 0 // 0 = majority, 1 = consensus
val k = 1

val hte_bd_model = new HTE_BD(noisyData, nTrees, partitions, vote, k, maxDepthRF, 48151623)

val hte_bd = hte_bd_model.runFilter()

hte_bd.persist()
hte_bd.count()

trainDT(hte_bd, test, 20)

trainKNN(hte_bd, test)


// HTE_BD Clean Data

import org.apache.spark.mllib.feature._

val nTrees = 100
val maxDepthRF = 10
val partitions = 4
val vote = 0 // 0 = majority, 1 = consensus
val k = 1

val hte_bd_model_clean = new HTE_BD(train, nTrees, partitions, vote, k, maxDepthRF, 48151623)

val hte_bd_clean = hte_bd_model_clean.runFilter()

hte_bd_clean.persist()
hte_bd_clean.count()


trainDT(hte_bd_clean, test, 20)

trainKNN(hte_bd_clean, test)
