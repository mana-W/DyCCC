#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)

input_rds <- args[[1]]
output_csv <- args[[2]]
top_n <- if (length(args) >= 3) as.integer(args[[3]]) else 100L

mat <- readRDS(input_rds)
if (inherits(mat, "dgCMatrix") || inherits(mat, "matrix")) {
  df <- as.data.frame(as.matrix(mat))
  df$target <- rownames(df)
  long <- reshape(
    df,
    varying = setdiff(colnames(df), "target"),
    v.names = "weight",
    timevar = "ligand",
    times = setdiff(colnames(df), "target"),
    idvar = "target",
    direction = "long"
  )
  long <- long[long$weight > 0, c("ligand", "target", "weight")]
} else if (is.data.frame(mat)) {
  long <- mat
  names(long) <- tolower(names(long))
  if (!all(c("ligand", "target", "weight") %in% names(long))) {
    stop("Data frame RDS must contain ligand,target,weight columns")
  }
  long <- long[, c("ligand", "target", "weight")]
} else {
  stop("Unsupported RDS object.")
}

long <- long[!is.na(long$ligand) & !is.na(long$target) & !is.na(long$weight), ]
long <- long[order(long$ligand, -long$weight), ]
long <- do.call(rbind, lapply(split(long, long$ligand), function(x) head(x, top_n)))
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
write.csv(long, output_csv, row.names = FALSE)
