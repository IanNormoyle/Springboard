Q1
SELECT name from Facilities WHERE membercost > 0;
Q2
SELECT COUNT(*) from Facilities WHERE membercost > 0;
Q3
SELECT COUNT(*) from Facilities WHERE membercost < .2 * monthlymaintenance;
Q4
SELECT * from Facilities WHERE facid = 1 OR facid = 5;
Q5
SELECT name, monthlymaintenance, CASE WHEN monthlymaintenance > 100 then "Expensive" Else "Cheap" end as "Expense" FROM Facilities as “Cost”;
Q6
SELECT firstname, surname FROM (SELECT firstname, surname, joindate, ROW_Number() over (order by joindate DESC) as "JoinRow" FROM Members) WHERE JoinRow = 1
Q7
SELECT * from Facilities WHERE facid = 1 UNION SELECT * from Facilities WHERE facid = 5
SELECT * from Facilities WHERE facid IN (1,5)
Q8
SELECT DISTINCT Concat(Members.firstname,' ',Members.surname) as "Fullname",Facilities.name
From Members
JOIN Bookings ON Members.memid = Bookings.memid 
JOIN Facilities ON Bookings.facid = Facilities.facid
WHERE Bookings.facid = 1 OR Bookings.facid = 0
Q9
SELECT CONCAT(Members.firstname, ' ', Members.surname) AS "fullname",
       Facilities.name,
       CASE
           WHEN Bookings.memid > 0 THEN Facilities.membercost * Bookings.slots
           ELSE Facilities.guestcost * Bookings.slots
       END AS "UseCost"
FROM Bookings
JOIN Facilities ON Bookings.facid = Facilities.facid
JOIN Members ON Bookings.memid = Members.memid
WHERE DATE(starttime) = '2012-09-04' AND (CASE
           WHEN Bookings.memid > 0 THEN Facilities.membercost * Bookings.slots
           ELSE Facilities.guestcost * Bookings.slots END) > 30 ORDER By UseCost DESC;
Q10
SELECT
name,
revenue
	FROM
		(SELECT
			Facilities.name, SUM(CASE WHEN Bookings.memid = 0 THEN Facilities.guestcost * Bookings.slots ELSE Facilities.membercost * Bookings.slots END) AS revenue
	FROM Bookings INNER JOIN Facilities
	ON Bookings.facid = Facilities.facid
	GROUP BY name) AS inner_table
	WHERE revenue < 1000
ORDER BY revenue;
Q11
SELECT Members.surname, Members.firstname, Members.recommendedby 
	FROM Members
	ORDER BY Members.surname DESC
Q12
SELECT CONCAT(Bookings.slots, Facilities.name) as Member, Bookings.memid
	FROM Bookings 
	JOIN Facilities ON Bookings.facid = Facilities.facid
	WHERE Bookings.memid != 0;
Q13
SELECT Bookings.slots, Facilities.name, Bookings.memid, Bookings.starttime
FROM Bookings 
JOIN Facilities ON Bookings.facid = Facilities.facid
WHERE Bookings.memid != 0
ORDER BY Bookings.starttime;
