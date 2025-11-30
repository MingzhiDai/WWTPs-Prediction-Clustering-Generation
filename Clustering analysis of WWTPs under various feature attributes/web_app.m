function web_app
    % 创建窗口
    fig = uifigure('Name','Clustering analysis under different WWTP feature attributes',...
                   'Position',[350 100 500 650]);

    % 标题
    uilabel(fig,'Text','WWTP DPNG-EPMC clustering analysis',...
        'Position',[80 610 400 30],...
        'FontSize',16,'FontWeight','bold');

    %-----------------------------
    % 输入区域
    %-----------------------------
    y = 560;
    uilabel(fig,'Text','Input_File','Position',[50 y+25 200 20]);
    Input_File = uieditfield(fig,'text','Position',[50 y 400 25],...
        'Value','D:\BaiduNetdiskDownload\clustering\WWTP--environmental and geographical features\Phylum\WWTP_Phylum--environmental and geographical features.csv');

    y = y - 60;
    uilabel(fig,'Text','Microbial_hierarchy','Position',[50 y+25 200 20]);
    Microbial_hierarchy = uieditfield(fig,'text','Position',[50 y 400 25],...
        'Value','Phylum');

    y = y - 60;
    uilabel(fig,'Text','Feature_attribute','Position',[50 y+25 200 20]);
    Feature_attribute = uieditfield(fig,'text','Position',[50 y 400 25],...
        'Value','environmental and geographical');

    y = y - 60;
    uilabel(fig,'Text','Microbial_feature_number','Position',[50 y+25 200 20]);
    Microbial_feature_number = uieditfield(fig,'numeric','Position',[50 y 400 25],'Value',21,'HorizontalAlignment','left');

    y = y - 60;
    uilabel(fig,'Text','Output_folder','Position',[50 y+25 200 20]);
    Output_folder = uieditfield(fig,'text','Position',[50 y 400 25],...
        'Value','D:\BaiduNetdiskDownload\clustering\WWTP--environmental and geographical features\Phylum');

    %-----------------------------
    % Run 按钮
    %-----------------------------
    y = y - 80;
    btn = uibutton(fig,'push','Text','Run clustering analysis',...
        'FontSize',14,'FontWeight','bold',...
        'Position',[150 y 200 40],...
        'ButtonPushedFcn',@(btn,event) run_main);

    %-----------------------------
    % 输出显示区
    %-----------------------------
    txt = uitextarea(fig,'Position',[50 30 400 150],'Editable','off');

    %======================================================================
    % 回调函数：运行 DPNGEPMCmain_paper.m
    %======================================================================
    function run_main
        txt.Value = "Running...";

        dlg = uiprogressdlg(fig,'Title','Running','Indeterminate','on',...
            'Message','Please wait...');

        try
            % 调用主程序
            DPNGEPMCmain_paper( ...
                Input_File.Value, ...
                Microbial_hierarchy.Value, ...
                Feature_attribute.Value, ...
                Microbial_feature_number.Value, ...
                Output_folder.Value ...
                );

            txt.Value = "Run finished.";
            multiFileDownloader(Output_folder.Value, Microbial_hierarchy.Value, Feature_attribute.Value)
        catch ME
            txt.Value = ["Error occurred:", ME.message];
        end
        close(dlg);
    end
end
