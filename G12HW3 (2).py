from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
from collections import Counter
import threading
import time
import sys
import os
import math
import numpy as np
import random as rand
import statistics


T = -1      #To be set via command line
p = 8191    #Values used to generate hash functions. Created in order to change it if wanted
D = -1
W = -1      #To be set via command line


#Function generating parameters for define hash functions
def hash_function_generator( p ):
      a = rand.randint(1, p-1) #Randomly create a
      b = rand.randint(0, p-1 ) #Same but for b

      return (a, b) #return two parameters

#Function generating output of a hash function given its parameters
def result_hash(a, b, number_columns, item, p):
      
      return (((a * item) + b) % p) % number_columns


#Function generating output of a binary hash function given its parameters
def binary_result_hash(a, b, item, p):

      result = (((a * item) + b) % p) % 2

      if result == 0:
           return -1
      else:
           return 1
      

#Function that calculates for each list of elemnts its partial sketch and partial exact frequencies
def partial_sketches_computation(pair, hash_functions, binary_hash_functions):
     
      partial_exact_frequencies = {} #Dictionary where our partial exact frequencies will be stored
      partial_CM = np.zeros((D, W), dtype=int)  #Array where our partial count-min sketches will be stored
      partial_CS = np.zeros((D, W), dtype=int) #Array where our partial count sketch will be stored

      _, items = pair #Separate random key and list of numbers

      for element in items: #Iterate through each element of the list
          
            if element in partial_exact_frequencies: #Check if we already found the values
                  partial_exact_frequencies[element] = partial_exact_frequencies[element] + 1 #If yes, sum 1
            else:
                  partial_exact_frequencies[element] = 1 #Insert the item in the dictionary

            for i in range(D): #Iterate through each row of the sketch
                  hash_parameters = hash_functions[i] #Get i-th hash function, which is associated to row i of the sketch
                  binary_hash_parameters = binary_hash_functions[i] #Same as before, but for the binary hash functions
                  index_column = result_hash(hash_parameters[0], hash_parameters[1], W, element, p) #Get the column retrieved by the hashing of the element

                  partial_CM[i][index_column] = partial_CM[i][index_column] + 1 #Update according to the count-min sketch algorithm
                  partial_CS[i][index_column] = partial_CS[i][index_column] + binary_result_hash(binary_hash_parameters[0], binary_hash_parameters[1], element, p) #Update according to count sketch algorithm
                 

            
      return (1, (partial_exact_frequencies, partial_CM, partial_CS))
            

#Pairs has the following shape : (1, list of tuples(partial_freq_dict, partial_CM, partial_CS))
def final_sketches_computation(pair):
     
      _, items = pair #Separate temp key and list of tuples
      counter_exact_frequencies = Counter() #Dictionary where our final exact frequencies will be stored
      final_CM = np.zeros((D, W), dtype=int)  #Array where our final count-min sketches will be stored
      final_CS = np.zeros((D, W), dtype=int) #Array where our final count sketch will be stored

      for element in items:
            #Transform the dictionary in a object of kind "Counter"
            counter_dict = Counter(element[0])
            counter_exact_frequencies = counter_exact_frequencies + counter_dict #Sum the dictionaries
            final_CM = final_CM + element[1]
            final_CS = final_CS + element[2]
      
      
      final_exact_frequencies = dict(counter_exact_frequencies)

      return (final_exact_frequencies, final_CM, final_CS)



def compute_frequencies_for_k_hitters(elements, CM, CS, hash_functions, binary_hash_functions):

      final_list = []

      for e in elements: #Iterate through each element of the list, where an element is of the kind (int, frequency)

            possible_frequencies_CM = []
            possible_frequencies_CS = []

            for i in range(D): #Iterate through each row
                  hash_parameters = hash_functions[i] #Get function of row i
                  binary_hash_parameters = binary_hash_functions[i] #Same but for binary hash function
                  index_column = result_hash(hash_parameters[0], hash_parameters[1], W, e[0], p) #Get index column of mapping
                  possible_frequencies_CM.append(CM[i][index_column])
                  possible_frequencies_CS.append( binary_result_hash(binary_hash_parameters[0], binary_hash_parameters[1], e[0], p) * CS[i][index_column])

            #Compute estimated frequency from both sketches
            estimate_frequency_CM = min(possible_frequencies_CM)
            estimate_frequency_CS = statistics.median(possible_frequencies_CS)
                  
            #Compute error for both sketches
            error_CM = abs( e[1] - estimate_frequency_CM ) / e[1]
            error_CS = abs( e[1] - estimate_frequency_CS ) / e[1]

            #Append new element in the list that will be returned
            new_element = ( e[0], e[1], (estimate_frequency_CM, error_CM), (estimate_frequency_CS, error_CS))
            final_list.append(new_element)
                  

      return final_list

                  

                       
def process_batch(time, batch):

      global streamLength, exact_frequencies_dict, CS, CM, hash_functions, binary_hash_functions

      batch_size = batch.count() #Get size of the batch

      if streamLength[0] >= T:  #Check if we processed alread T elements
        return
      streamLength[0] += batch_size #Update stream element

      random_key = int(math.sqrt(batch_size)) #Maximum value which a random key can take
       
      batch_sketches = (batch.map(lambda x: (rand.randint(0, random_key - 1), int(x))) #MAP1
                              .groupByKey() #SHUFFLING
                              .map(lambda x: partial_sketches_computation(x, hash_functions, binary_hash_functions)) #REDUCE 1
                              .groupByKey() #SHUFFLING
                              .map(lambda x: final_sketches_computation(x)) #REDUCE 2
                              ) 


      results = batch_sketches.collect() #Collect results = [(dict, cm, cs)]

      if results: #If we have processed a batch, update global structures for frequency estimation
            exact_frequencies_dict = dict(Counter(exact_frequencies_dict) + Counter(results[0][0]))
            CM = CM + results[0][1]
            CS = CS + results[0][2]
           



      if streamLength[0] >= T:  #If it is greater than the threshold, stop 
            stopping_condition.set()
      
      
      
      
      
      


if __name__ == "__main__":
      # Checking number of cmd line parameters
      assert len(sys.argv) == 6, "User must provide  port, threshold, rows, columns and number of items desired"

      # Spark setup
      conf = SparkConf().setMaster("local[*]").setAppName("Homework 3")
      sc = SparkContext(conf=conf)
      ssc = StreamingContext(sc, 0.01)  #Batch duration of 0.01 seconds
      ssc.sparkContext.setLogLevel("ERROR") 

      stopping_condition = threading.Event()
    

      # Input Reading 

      # 1. Read port number
      portExp = sys.argv[1]
      assert portExp.isdigit(), "Port number must be an integer" 
      portExp = int(portExp)

      # 2. Read maximum number of items to process
      T = sys.argv[2]
      assert T.isdigit(), "T must be an integer"
      T = int(T)

      # 3. Read number of rows of each sketch
      D = sys.argv[3]
      assert D.isdigit(), "D must be an integer"
      D = int(D)

      # 4. Read number of columns of each sketch
      W = sys.argv[4]
      assert W.isdigit(), "W must be an integer"
      W = int(W)

      # 5. Read number of top frequent items of interest
      K = sys.argv[5]
      assert K.isdigit(), "K must be an integer"
      K = int(K)


      streamLength = [0] #Used to count how many items have been processed 
      exact_frequencies_dict = {} #Array where exact frequencies will be stored

      #Lists which will contain our hash functions
      hash_functions = [0] * D 
      binary_hash_functions = [0] * D

      CM = np.zeros((D, W), dtype=int)  #Array where our count-min sketch will be stored
      CS = np.zeros((D, W), dtype=int) #Array where our count sketch will be stored

      for i in range(D):
            hash_functions[i] = hash_function_generator(p)
            binary_hash_functions[i] = hash_function_generator(p)

      stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)   #Connect stream coming from indicated server and port


      stream.foreachRDD(lambda time, batch: process_batch(time, batch))   #Function indicating how each batch will be processed


      #print("Starting streaming engine")
      ssc.start()
      #print("Waiting for shutdown condition")
      stopping_condition.wait()
      #print("Stopping the streaming engine")


      ssc.stop(False, False)
      #print("Streaming engine stopped")

      exact_frequencies_dict = dict(sorted(exact_frequencies_dict.items(), key=lambda item: item[1], reverse=True)) #Sort dictionary in non-increasing order. This is done in order to retrieve top-k heavy hitters fastly
      K_frequency = 0

      #For loop used to retrieve the true frequency of the K-th element 
      for i, (k, v) in enumerate(exact_frequencies_dict.items()):
            if i == K-1:
                  K_frequency = v

      top_k_heavy_hitters = []  #Here we will store top-k heavy hitters

      for item, freq in exact_frequencies_dict.items(): #Iterate through elements of dictionary
            if freq < K_frequency: #Once you found an element which is not a top-k heavy hitter
                  break
            top_k_heavy_hitters.append((item, freq)) #Append it


      #Compute frequencies and error for each one of the heavy hitters
      results = compute_frequencies_for_k_hitters( top_k_heavy_hitters, CM, CS, hash_functions, binary_hash_functions)


      average_CM_error = 0
      average_CS_error = 0

      number_k_hitters = len(results) #Number of top-k Heavy hitters

      for e in results:
            average_CM_error = average_CM_error + e[2][1]
            average_CS_error = average_CS_error + e[3][1]
      
      average_CM_error = average_CM_error/ number_k_hitters
      average_CS_error = average_CS_error/ number_k_hitters

      print(f"Port = {portExp} T = {T} D = {D} W = {W} K = {K}")
      print(f"Number of processed items = {streamLength[0]}")
      print(f"Number of distinct items = {len(exact_frequencies_dict.items())}")
      print(f"Number of Top-K Heavy Hitters = {number_k_hitters}")
      print(f"Avg Relative Error for Top-K Heavy Hitters with CM = {average_CM_error}")
      print(f"Avg Relative Error for Top-K Heavy Hitters with CS = {average_CS_error}")


      if number_k_hitters <= 10:
            results.sort(key=lambda x: x[0])  #sort in increasing order of item

            print(f"Top-K Heavy Hitters:")

            for e in results:
                  print(f"Item {e[0]} True Frequency = {e[1]} Estimated Frequency with CM = {e[2][0]}")





