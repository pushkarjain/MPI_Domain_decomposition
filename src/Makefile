CC = mpicc
OUT = out
COMMON = domain.c
BLOCK_SRC = block.c $(COMMON)
ROW_SRC = row.c $(COMMON)


block : 
	${CC} $(BLOCK_SRC) -o $(OUT)

row : 
	${CC} $(ROW_SRC) -o $(OUT)

clean : 
	rm -rf *.o outfile.txt $(OUT)

help : 
	@echo "make block - Domain decomposition by blocks"
	@echo "make row - Domain decomposition by rows"
	@echo "Usage: "
	@echo "./$(OUT)"

