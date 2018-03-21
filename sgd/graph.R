wd = getwd()
setwd("~/code/stat672new/sgd/img/")
png("comparison.png", 
    height = 300,
    width = 300)
curve(x * log(1 / 0.05),
      0, 
      100,
      col = "red",
      main = "Time to (fixed) accuracy",
      lwd = 2,
      xlab = "n",
      ylab = "Time")
abline(h=1/0.05,
       col = "blue",
       lwd = 2)
legend("topleft",
       lwd = 2,
       col = c("red", "blue"),
       legend = c("GD", "SGD"))
dev.off()
setwd(wd)
