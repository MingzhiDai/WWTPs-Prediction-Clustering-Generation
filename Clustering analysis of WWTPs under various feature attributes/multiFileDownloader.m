function multiFileDownloader(Output_folder, Microbial_hierarchy, Feature_attribute)
    fig = uifigure('Name','文件下载中心','Position',[200 200 500 300]);

    files = [fullfile(Output_folder, ['WWTP_' Microbial_hierarchy '--' Feature_attribute '-Information of each centroid.xlsx'])... 
        fullfile(Output_folder, ['WWTP_' Microbial_hierarchy '--' Feature_attribute '-The WWTP ranking information of each centroid.xlsx'])... 
        fullfile(Output_folder, ['WWTP_' Microbial_hierarchy '--' Feature_attribute '-Microbio heatmap'])]; % 你自己的文件列表

    y = 240;
    for i = 1:length(files)
        filename = files(i);

        % 标签显示文件名
        uilabel(fig, ...
            'Text', filename, ...
            'Position', [30 y 200 25], ...
            'FontSize', 14);

        % "下载"按钮（点击才会询问保存路径）
        uibutton(fig, ...
            'Text', '下载', ...
            'Position', [250 y 80 25], ...
            'ButtonPushedFcn', @(btn,event)downloadFile(filename));

        y = y - 40;  % 下一行
    end
end

function downloadFile(filename)
    if ~isfile(filename)
        uialert(gcbf, ['文件不存在：' char(filename)], '错误');
        return;
    end

    % 用户点击按钮后，才弹出保存对话框
    [f, p] = uiputfile(filename, ['保存文件：' char(filename)]);
    if isequal(f,0)
        return;   % 用户取消，不保存
    end

    copyfile(filename, fullfile(p, f));
    uialert(gcbf, '文件已成功保存。','成功');
end
