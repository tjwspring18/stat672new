df = data.frame(
		n = c(10, 20, 50, 100, 200, 500, 1000),
		emp_error_1d = c(0.3, 0.345, 0.402, 0.4512, 0.4297, 0.4244, 0.4454),
		gen_error_1d = c(0.476, 0.461, 0.4948, 0.468, 0.4673, 0.45388, 0.45954),
		emp_error_2d = c(0.142, 0.16, 0.1832, 0.169, 0.1825, 0.1758, 0.17388),
		gen_error_2d = c(0.268, 0.276, 0.2208, 0.1868, 0.1932, 0.18252, 0.17602),
		emp_error_5d = c(0.108, 0.144, 0.1812, 0.181, 0.1804, 0.1754, 0.17966),
		gen_error_5d = c(0.27, 0.236, 0.2056, 0.2028, 0.1927, 0.18176, 0.18154)
		)

png("plot.png", height = 3, width=6, units = "in", res = 300)
par(mfrow=c(1,3))
plot(df$emp_error_1d, 
     col = "black", 
     lty = 1, 
     lwd = 2,
     type="l", 
     ylim = c(0, 0.5),
     ylab = "Error",
     main = "d=1",
     xaxt = 'n', 
     xlab = "N")
lines(df$gen_error_1d, col = "black", lty = 2, lwd=2)
axis(1, at=1:7, labels = c(10, 20, 50, 100, 200, 500, 1000))
legend("bottomright", legend = c("emp_error", "gen_error"), lty = c(1,2), col =
       "black", lwd = 2)

plot(df$emp_error_2d, 
     col = "blue", 
     lty = 1, 
     lwd = 2,
     type="l", 
     ylim = c(0, 0.5),
     ylab = "Error",
     main = "d=2",
     xaxt = 'n', 
     xlab = "N")
lines(df$gen_error_2d, col = "blue", lty = 2, lwd=2)
axis(1, at=1:7, labels = c(10, 20, 50, 100, 200, 500, 1000))
legend("bottomright", legend = c("emp_error", "gen_error"), lty = c(1,2), col =
       "blue", lwd = 2)

plot(df$emp_error_5d, 
     col = "red", 
     lty = 1, 
     lwd = 2,
     type="l", 
     ylim = c(0, 0.5),
     ylab = "Error",
     main = "d=5",
     xaxt = 'n', 
     xlab = "N")
lines(df$gen_error_5d, col = "red", lty = 2, lwd=2)
axis(1, at=1:7, labels = c(10, 20, 50, 100, 200, 500, 1000))
legend("bottomright", legend = c("emp_error", "gen_error"), lty = c(1,2), col =
       "red", lwd = 2)

dev.off()
