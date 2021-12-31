SELECT
    a.*,
    b.*
FROM
    samples_oso2018_T31TCL AS a
    samples_oso2019_T31TCL AS b
WHERE
    a.code = n
    AND
    b.code = n
    AND
    ST_Intersects(a.geometry, b.geometry)