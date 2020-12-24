function plotResults(fileName)
    data = readtable("data/"+fileName+".csv");

    f = figure;
    a = axes(f);
    t = 0:height(data.Returns)-1;
    errorbar(a, t, data.Returns, data.Error)
    
    prefix = extractBetween(fileName, "", "_out");
    title("Obstructed Gridworld: " + prefix)
    
    xlabel("Episode")
    ylabel("Average Return")
    legend(""+sum(data.Returns)/1000, ...
        "Location","BEST")

    exportgraphics(f, "plots/"+prefix+"_plot.png");
end