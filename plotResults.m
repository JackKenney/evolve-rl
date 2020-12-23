data = readtable("out.csv");

f = figure;
a = axes(f);
t = 0:height(data.BBO)-1;
errorbar(a, t, data.BBO, data.BBOErrorBar)
hold on
errorbar(a, t, data.REINFORCE, data.REINFORCEErrorBar)
hold off

title("Obstructed Gridworld")
xlabel("Episode")
ylabel("Average Return")
legend([ ...
    "BBO    "+sum(data.BBO)/1000, ...
    "REINFORCE "+sum(data.REINFORCE)/1000 ...
    ], ...
    "Location","BEST")

exportgraphics(f, "plot.png");