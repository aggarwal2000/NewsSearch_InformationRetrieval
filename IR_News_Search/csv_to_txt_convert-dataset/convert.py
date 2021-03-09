import csv 


def read_data(file,num):
    
	row_count = 0  
	with open(file,"r",encoding="utf-8",errors = "ignore") as csvfile:
        
		csvreader = csv.reader(csvfile)
        
		while(True):

			try:
				for row in csvreader:

					if(row_count == 0):
						headers = row
					
					if(row_count >= 1):
						try:
							print("int:",int(row[1]))
							f = open("../news_docs/" + row[1] + ".txt", "w",encoding="utf-8")
							print("doc_id:",row[1])
						
							for i in range(2,10):
								#print(headers[i])
								
								f.write(headers[i])
								f.write(":\n")
								f.write(row[i])
								f.write("\n\n")
								#print(row[i])
								#print("\n")
							print("----------------------------------------")
							f.close()
						except:
							pass
						

					row_count +=1 

					if(row_count > num):
						break

			except:
				continue

			
			break


			
	


read_data("articles1.csv",52000)
read_data("articles2.csv",50200)
read_data("articles3.csv",100)

           