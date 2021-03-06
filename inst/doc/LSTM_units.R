## ----feedforward_plot,echo=F,fig.height=4,fig.width=7-------------------------
par(mar=rep(0,4))
plot(c(0,10),c(0,10),type="n",
     # bty ="n",
     xlab ="", ylab= "",xaxt="n",yaxt="n")

text(x=0,y=9.75,labels="LSTM unit feed forward",pos=4,cex=2)
symbols(x=5,y=5,rectangles = matrix(8,nrow=1,ncol=2),inches = F,add = T) # unit itself
text(x=c(0,0,0.5,10,10,9.25),y=c(8,1.5,0,8,1.5,10),
     labels = c(expression('C'[t-1]),expression('h'[t-1]),expression('x'[t]),expression('C'[t]),expression('h'[t]),expression('y'[t])))

symbols(x=c(1.5,2, 3,3.5,4.5,5,6,6.5),y=rep(2.5,8),rectangles = matrix(c(0.45,0.6),nrow=8,ncol=2,byrow = T),inches = F,add = T) # weight matrix
text(x=c(1.5,2, 3,3.5,4.5,5,6,6.5),y=rep(2.5,8),
     labels = c(expression('W'[f]),expression('U'[f]),expression('W'[i]),expression('U'[i]),
                expression('W'[g]),expression('U'[g]),expression('W'[o]),expression('U'[o]))
     ) # weight matrix

symbols(x=c(1.75,3.25,4.75,6.25),y=rep(4,4),circles = rep(0.15,4),inches = F,add = T) # addition of the weight matrix from x and hidden
text(x=c(1.75,3.25,4.75,6.25),y=rep(4,4),labels="+") # addition of the weight matrix from x and hidden

symbols(x=c(1.75,3.25,4.75,6.25,8),y=c(rep(5,4),7),rectangles = matrix(c(1,0.5),nrow=5,ncol=2,byrow = T),inches = F,add = T) # linearisation
text(x=c(1.75,3.25,4.75,6.25,8),y=c(rep(5,4),7),labels=c("sigmoid","sigmoid","tanh","sigmoid","tanh")) # linearisation

symbols(x=c(1.75,4.75,4.75,8),y=c(8,8,7,6),circles = rep(0.15,4),inches = F,add = T) # addition of the weight matrix from x and hidden
text(x=c(1.75,4.75,4.75,8),y=c(8,8,7,6),labels=c("x","+","x","x")) # addition of the weight matrix from x and hidden
# arrows(x0)

segments(x0=c(1),x1 = c(6.5),y0 = c(0),y1 = c(0)) # xt segment
segments(x0=c(0.5),x1 = c(6),y0 = c(1.5),y1 = c(1.5)) # ht-1 segment

arrows(x0=c(2, 3.5,5,6.5),y0=rep(0,4),y1=rep(2.2,4),length = 0.1) # xt arrows
arrows(x0=c(1.5, 3,4.5,6),y0=rep(1.5,4),y1=rep(2.2,4),length = 0.1) # ht-1 arrows

arrows(x0=c(1.5,2, 3,3.5,4.5,5,6,6.5),x1=c(1.5,2, 3,3.5,4.5,5,6,6.5)+c(0.1,-0.1),y0=rep(2.8,8),y1=rep(3.85,8),length = 0.1) # before addition arrows

arrows(x0=c(1.75,3.25,4.75,6.25),x1=c(1.75,3.25,4.75,6.25),y0=rep(4.3,4),y1=rep(4.75,4),length = 0.1) # before linearisation

arrows(x0=c(0.5,1.9,4.9),x1=c(1.6,4.6,9.75),y0=rep(8,3),length = 0.1) # C t-1 to Ct arrows

arrows(x0=c(1.75,4.75),y0=rep(5.25,2),y1=c(7.75,6.75),length = 0.1)# after linearisation
segments(x0=c(3.25,6.25),y0=rep(5.25,2),y1=c(7,6))# after linearisation

arrows(x0=c(3.25,6.25),x1=c(4.55,7.8),y0=c(7,6),length = 0.1) # still after linearisation

arrows(x0=c(4.75),y0=c(7.3),y1=c(7.7),length = 0.1) # still still after linearisation

arrows(x0=c(8),y0=c(8),y1=c(7.25),length = 0.1) # before C tanh
arrows(x0=c(8),y0=c(6.75),y1=c(6.3),length = 0.1) # after C tanh
segments(x0=c(8),y0=c(5.7),y1=c(1.5)) # after X
arrows(x0=c(8),x1 = c(9.75),y0=c(1.5),length = 0.1) # after X
arrows(x0=c(9.25),y0=c(1.5),y1=c(9.5),length = 0.1) # after X

text(x=c(1.75,3.25,4.75,6.25),y=5.5,labels=c(expression('f'[t]),expression('i'[t]),expression('g'[t]),expression('o'[t])),pos = 4)

## ----BPTT plot,fig.height=5,fig.width=7,echo=F--------------------------------
par(mar=rep(0,4))
plot(c(0,10),c(0,10),type="n",
     # bty ="n",
     xlab ="", ylab= "",xaxt="n",yaxt="n")

text(x=0,y=9.75,labels="LSTM unit back propagation",pos=4,cex=2)
symbols(x=5,y=5,rectangles = matrix(8,nrow=1,ncol=2),inches = F,add = T) # unit itself
text(x=c(0,0,0.5,10,10,9.25),y=c(8,1.5,0,8,1.5,10),
     labels = c(expression('C'[t-1]),expression('h'[t-1]),expression('x'[t]),expression('C'[t]),expression('h'[t]),expression('y'[t])))

symbols(x=c(1.5,2, 3,3.5,4.5,5,6,6.5),y=rep(2.5,8),rectangles = matrix(c(0.45,0.6),nrow=8,ncol=2,byrow = T),inches = F,add = T) # weight matrix
text(x=c(1.5,2, 3,3.5,4.5,5,6,6.5),y=rep(2.5,8),
     labels = c(expression('W'[f]),expression('U'[f]),expression('W'[i]),expression('U'[i]),
                expression('W'[g]),expression('U'[g]),expression('W'[o]),expression('U'[o]))
     ) # weight matrix

symbols(x=c(1.75,3.25,4.75,6.25),y=rep(4,4),circles = rep(0.15,4),inches = F,add = T) # addition of the weight matrix from x and hidden
text(x=c(1.75,3.25,4.75,6.25),y=rep(4,4),labels="+") # addition of the weight matrix from x and hidden

symbols(x=c(1.75,3.25,4.75,6.25,8),y=c(rep(5,4),7),rectangles = matrix(c(1,0.5),nrow=5,ncol=2,byrow = T),inches = F,add = T) # linearisation
text(x=c(1.75,3.25,4.75,6.25,8),y=c(rep(5,4),7),labels=c("sigmoid","sigmoid","tanh","sigmoid","tanh")) # linearisation

symbols(x=c(1.75,4.75,4.75,8),y=c(8,8,7,6),circles = rep(0.15,4),inches = F,add = T) # addition of the weight matrix from x and hidden
text(x=c(1.75,4.75,4.75,8),y=c(8,8,7,6),labels=c("x","+","x","x")) # addition of the weight matrix from x and hidden
# arrows(x0)

arrows(x1=c(1),x0 = c(6.5),y0 = c(0),y1 = c(0),length = 0.1) # xt segment
arrows(x1=c(0.5),x0 = c(6),y0 = c(1.5),y1 = c(1.5),length = 0.1) # ht-1 segment

arrows(x0=c(2, 3.5,5,6.5),y1=rep(0,4),y0=rep(2.2,4),length = 0.1) # xt arrows
arrows(x0=c(1.5, 3,4.5,6),y1=rep(1.5,4),y0=rep(2.2,4),length = 0.1) # ht-1 arrows

arrows(x1=c(1.5,2, 3,3.5,4.5,5,6,6.5),x0=c(1.5,2, 3,3.5,4.5,5,6,6.5)+c(0.1,-0.1),y1=rep(2.8,8),y0=rep(3.85,8),length = 0.1) # before addition arrows

arrows(x0=c(1.75,3.25,4.75,6.25),y1=rep(4.3,4),y0=rep(4.75,4),length = 0.1) # before linearisation

arrows(x1=c(0.5,1.9,4.9),x0=c(1.6,4.6,9.75),y0=rep(8,3),length = 0.1) # C t-1 to Ct arrows

arrows(x0=c(1.75,4.75),y1=rep(5.25,2),y0=c(7.75,6.75),length = 0.1)# after linearisation
arrows(x0=c(3.25,6.25),y1=rep(5.25,2),y0=c(7,6),length = 0.1)# after linearisation

segments(x1=c(3.25,6.25),x0=c(4.55,7.8),y0=c(7,6)) # still after linearisation

arrows(x0=c(4.75),y1=c(7.3),y0=c(7.7),length = 0.1) # still still after linearisation

arrows(x0=c(8),y1=c(8),y0=c(7.25),length = 0.1) # before C tanh
arrows(x0=c(8),y1=c(6.75),y0=c(6.3),length = 0.1) # after C tanh
arrows(x0=c(8),y1=c(5.7),y0=c(1.5),length = 0.1) # after X
segments(x1=c(8),x0 = c(9.75),y0=c(1.5)) # after X
arrows(x0=c(9.25),y1=c(1.5),y0=c(9.5),length = 0.1) # after X

text(x=c(1.75,3.25,4.75,6.25),y=5.5,labels=c(expression('f'[t]),expression('i'[t]),expression('g'[t]),expression('o'[t])),pos = 4)

text(x = c(9.25), y = c(9.25), labels=c(expression(delta~"1")), pos=4,cex=0.8)
text(x = c(9.75), y = c(8), labels=c(expression(delta~"4")), pos=3,cex=0.8)
text(x = c(9.75), y = c(1.5), labels=c(expression(delta~"2")), pos=3,cex=0.8)
text(x = c(8), y = c(2), labels=c(expression(delta~"3")), pos=4,cex=0.8)
text(x = c(8), y = c(6.35), labels=c(expression(delta~"5")), pos=4,cex=0.8)
text(x = c(8), y = c(7.55), labels=c(expression(delta~"7")), pos=4,cex=0.8)
text(x = c(7), y = c(6), labels=c(expression(delta~"6")), pos=3,cex=0.8)
text(x = c(7), y = c(8), labels=c(expression(delta~"8")), pos=3,cex=0.8)
text(x = c(3), y = c(8), labels=c(expression(delta~"8")), pos=3,cex=0.8)
text(x = c(4.75), y = c(7.55), labels=c(expression(delta~"8")), pos=4,cex=0.8)
text(x = c(4.75), y = c(6.25), labels=c(expression(delta~"9")), pos=4,cex=0.8)
text(x = c(3.25), y = c(6.25), labels=c(expression(delta~"10")), pos=4,cex=0.8)
text(x = 0.5, y = c(8), labels=c(expression(delta~"11")), pos=3,cex=0.8)
text(x = c(1.75), y = c(6.25), labels=c(expression(delta~"12")), pos=4,cex=0.8)

text(x = c(6.25), y = c(4.35), labels=c(expression(delta~"13")), pos=4,cex=0.8)
text(x = c(4.75), y = c(4.35), labels=c(expression(delta~"14")), pos=4,cex=0.8)
text(x = c(3.25), y = c(4.35), labels=c(expression(delta~"15")), pos=4,cex=0.8)
text(x = c(1.75), y = c(4.35), labels=c(expression(delta~"16")), pos=4,cex=0.8)

text(x = c(6.25)+0.2, y = c(4.35)-1, labels=c(expression(delta~"13")), pos=4,cex=0.8)
text(x = c(4.75)+0.2, y = c(4.35)-1, labels=c(expression(delta~"14")), pos=4,cex=0.8)
text(x = c(3.25)+0.2, y = c(4.35)-1, labels=c(expression(delta~"15")), pos=4,cex=0.8)
text(x = c(1.75)+0.2, y = c(4.35)-1, labels=c(expression(delta~"16")), pos=4,cex=0.8)

text(x = c(6.5)-0.1, y = c(0.5), labels=c(expression(delta~"17")), pos=4,cex=0.8)
text(x = c(6)-0.1, y = c(1.85), labels=c(expression(delta~"18")), pos=4,cex=0.8)
text(x = c(5)-0.1, y = c(0.5), labels=c(expression(delta~"19")), pos=4,cex=0.8)
text(x = c(4.5)-0.1, y = c(1.85), labels=c(expression(delta~"20")), pos=4,cex=0.8)
text(x = c(3.5)-0.1, y = c(0.5), labels=c(expression(delta~"21")), pos=4,cex=0.8)
text(x = c(3)-0.1, y = c(1.85), labels=c(expression(delta~"22")), pos=4,cex=0.8)
text(x = c(2)-0.1, y = c(0.5), labels=c(expression(delta~"23")), pos=4,cex=0.8)
text(x = c(1.5)-0.1, y = c(1.85), labels=c(expression(delta~"24")), pos=4,cex=0.8)
text(x = 0.5, y = 1.5, labels=c(expression(delta~"25")), pos=3,cex=0.8)
text(x = 1, y = 0, labels=c(expression(delta~"26")), pos=3,cex=0.8)

