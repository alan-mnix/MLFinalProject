data = load('hue');
n = size(data.Xtrain,1)

f = fopen('huel.txt', 'wt')
for i=1:n
	fprintf(f, '%d\n', data.Ytrain(i));
end
fclose(f)


f = fopen('hue.txt', 'wt')
fprintf(f, '%d\n', n);
for i=1:n
i
[r c] = find(data.Xtrain(i,:));
for j=c
	fprintf(f, '%d:%d ', j, data.Xtrain(i,j));
end
fprintf(f, '\n');
end

fclose(f);
