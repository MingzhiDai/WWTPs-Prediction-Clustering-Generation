function draw_clustering_heatmap(input_excel, Microbial_hierarchy, output_image)

% 绘制微生物 Phylum 热力图
% input_excel：Excel 文件路径
% output_image：输出图片路径

    %=== 读取数据 ===%
    df = readcell(input_excel);   % 用 readcell 可以同时读取字符串+数字
    data = cell2mat(df(2:end, 2:6));   % 取第 2~6 列作为数值（跳过表头）

    %=== 自动读取 y_ticks（第一列的分类名称） ===%
    y_ticks = df(2:end, 1);   % 读取 A2:A22，作为 21 个 Phylum 名称

    %=== x 轴标签（固定） ===%
    x_ticks = {'Cluster1','Cluster2','Cluster3','Cluster4','Cluster5'};

    %=== 绘图 ===%
    figure('Position',[100 100 800 700], 'Color','w');
    h = heatmap(data);

    % 设置参数
    h.Colormap = redcolormap();
    h.ColorLimits = [0 50];
    h.CellLabelFormat = '%.4f';

    h.XDisplayLabels = x_ticks;
    h.YDisplayLabels = y_ticks;

    h.YLabel = ['Different' '' Microbial_hierarchy 's'];
    h.FontSize = 12;

    %=== 保存图片 ===%
    exportgraphics(h, output_image, 'Resolution', 600);
end


% seaborn "Reds" colormap
function cmap = redcolormap()
    r = linspace(1, 0.6, 256)';   % 红色从亮到深
    g = linspace(0.95, 0.1, 256)'; % 绿色较低，避免变黄
    b = linspace(0.95, 0.1, 256)'; % 蓝色较低，保持红色系
    cmap = [r g b];
end
