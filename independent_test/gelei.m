function [zhengque] =gelei(test_label,predict_label,zongleibie,yangben )
zhengque=zeros(1,zongleibie);
%�����ǩ��yangben*��������ʽ
for j=1:zongleibie
    for i=1:yangben
        if  test_label(i,j)==1
            if  predict_label(i,j)==1
            zhengque(j)=zhengque(j)+1;
            end
    end
end
end



