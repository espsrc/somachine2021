val textFile = sc.textFile("file:///home/administrador/datasets/hamlet.txt")

val counts = textFile.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)
