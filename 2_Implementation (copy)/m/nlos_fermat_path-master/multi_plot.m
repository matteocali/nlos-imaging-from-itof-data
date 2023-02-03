figure;
    for i = 1 : discontDetectionPara.numOfDiscont
        pathThis = Z(:, i);
        pathThis2 = Z2(:, i);
        F = scatteredInterpolant(X, Y, pathThis(:));
        F2 = scatteredInterpolant(X2, Y2, pathThis2(:));
        Vq = F(Xq, Yq);
        Vq2 = F2(Xq2, Yq2);
        if min(Vq(:)) == max(Vq(:)) || isnan(min(Vq(:)))
            continue;
        end
        plot3(X, Y, pathThis(:), 'mo', 'MarkerSize', 3, 'MarkerFaceColor', 'm'); hold on;
        plot3(X2, Y2, pathThis2(:), 'mo', 'MarkerSize', 3, 'MarkerFaceColor', 'g');
        surf(Xq, Yq, Vq, 'FaceColor', 'b', 'FaceAlpha', 0.3);
        surf(Xq2, Yq2, Vq2, 'FaceColor', 'r', 'FaceAlpha', 0.3)
        title(sprintf('Surface %d', i));
        xlim([xMin xMax]);
        ylim([yMin yMax]);
        xlabel('x');
        ylabel('y');
        axis equal
    end