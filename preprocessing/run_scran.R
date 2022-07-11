# Normalize single cell counts using cell factor scales

# Remove all objects from the current workspace (R memory).
rm(list = ls())

#BiocManager::install("scran")
#BiocManager::install("scRNAseq")

suppressMessages(library(scran))
suppressMessages(library(Matrix))
suppressMessages(library(scater))
suppressMessages(library(scRNAseq))

args <- commandArgs(trailingOnly = TRUE)

# Load DGC matrix file and metadata
mtx_file <- args[1]
g_file <- args[2]
bc_file <- args[3]
ofile <- args[4]


# Transpose matrix for python - R connection
counts <- t(as.matrix(readMM(mtx_file)))

# Read cell names
bc <- read.csv(bc_file, header=F, col.names = 'Cell', stringsAsFactors = F)

# Read gene names
g <- read.csv(g_file,header=F,col.names = 'Gene', stringsAsFactors = F)

print("Data loaded")



# Create single cell experiment
sce <- SingleCellExperiment(assays = list(counts = counts))

# Store the gene names in this object
rownames(sce) <- g$Gene
rowData(sce) <- "Gene"

# Store the gene names in this object
colnames(sce) <- bc$Cell

print("SCE created")



# Calculate stats
qcstats <- perCellQCMetrics(sce)
qcfilter <- quickPerCellQC(qcstats)
sce <- sce[,!qcfilter$discard]

print("QC filter ")
summary(qcfilter$discard)



# Compute clusters
clusters <- quickCluster(sce)

# Compute factors
sce <- computeSumFactors(sce, clusters=clusters)

print("Size factors")
summary(sizeFactors(sce))

# Log
sce <- logNormCounts(sce)

print(sce)



# Save this gene matrix to a tsv file
sce_norm <- as.data.frame(logcounts(sce))

write.table(sce_norm,
			file=ofile,
			quote=FALSE,
			sep='\t', 
			col.names = NA)

saveRDS(sce, file=gsub('.csv', '.rds', ofile))

print("SCE norm saved")

# End script