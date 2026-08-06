# Qwen vehicle localization and perspective protocol

Analyze only the primary side-view vehicle. AI is responsible for identifying
the vehicle body, its facing direction, the tight crop, and the perspective
correction calibration. Python will detect and draw all red vehicle structure;
do not identify or return BED/CAB/HOOD regions, structure keypoints, outlines,
or polylines.

Coordinates are normalized from 0 through 1000 relative to the supplied image.

## Ignore existing annotations

Ignore all red outlines, dimension lines, arrows, numbers, labels, translucent
overlays, and grids. They are post-processing graphics, not vehicle pixels.

## Localization and perspective

Return exactly one JSON object without Markdown or commentary. It must contain
these keys and value types:

- `bbox_1000`: array of four measured numbers;
- `boundary_touch_points_1000`: object containing the measured `[x,y]` arrays
  `leftmost`, `rightmost`, and `topmost`;
- `perspective_quad_1000`: array of four measured `[x,y]` arrays;
- `wheel_centers_1000`: array of two measured `[x,y]` arrays;
- `wheel_contact_points_1000`: array of two measured `[x,y]` arrays;
- `body_chassis_line_1000`: array of two measured `[x,y]` arrays;
- `background_type`: `white`, `transparent`, or `environment`;
- `front`: `left`, `right`, or `unknown`.

No example coordinates are provided. Every number must be measured anew from
the supplied image. Never reuse memorized, conventional, placeholder, evenly
spaced, or previously seen coordinates.

`bbox_1000` must tightly contain every visible fixed vehicle part and both tyre
contact points while excluding background, unrelated vehicles, text, shadows,
annotations, and thin protrusions such as antennas, whip antennas, roof racks
that are not part of the fixed body, and radio masts. The top edge must sit on
the highest fixed body surface (roof, roof-mounted light bar, or cargo rack)
and must not extend up to an antenna tip.

`boundary_touch_points_1000` is mandatory. It identifies the exact visible
vehicle pixel touching each corrected crop side:

- `leftmost`: the leftmost fixed vehicle pixel, such as a bumper or tow hook;
- `rightmost`: the rightmost fixed vehicle pixel;
- `topmost`: the highest fixed body pixel, such as the roof, a fixed roof
  light bar, or a cargo rack. Do NOT use an antenna tip, whip antenna, or
  radio mast — these are thin protrusions, not fixed body surfaces.

Every point must lie on the real vehicle outline, not merely share the same x
or y coordinate. Use zero safety margin. Exclude shadows, grass, road, exhaust
smoke, open doors, antennas, annotations, and background. Python crops the
left, right, and top edges through these points after perspective correction.
The bottom edge is determined separately from the two tyre contact points.

`perspective_quad_1000` is an internal calibration quadrilateral, never a copy
of the bounding box. Order its points as top-left, top-right, bottom-right,
bottom-left. Build it from real longitudinal body lines plus real vertical
door/pillar seams so that rectification makes the wheel-center line and
longitudinal body lines horizontal and vertical panel seams vertical.

All four labels refer strictly to the supplied source image, not to
vehicle-front/vehicle-rear and not to the later normalized front-right image.
For a vehicle whose front is on image-left, do not pre-mirror or exchange the
left and right calibration sides; Python performs that horizontal flip only
after rectification. The source side with the smaller apparent wheel/body
scale is the far side and may need enlargement, while the larger apparent
side is the near side and may need reduction.

The four calibration points must represent a rectangle painted onto one
physical side plane of the vehicle:

- top and bottom must be matching longitudinal body lines of similar length;
- left and right must be matching door or pillar seams of similar height;
- never use `topmost`, the roof outline, hood slope, bumpers, tyre contacts, or
  the four outer vehicle extremes as calibration corners;
- never join a short roof segment to a full-width lower edge;
- if the wheels, rocker line, beltline, and panel seams are already horizontal
  and vertical, return a nearly rectangular internal quad. Do not invent
  perspective.

`wheel_centers_1000` contains exactly the rear and front wheel centers.
`wheel_contact_points_1000` contains the two exact lowest visible tyre-rubber
points, one below each corresponding wheel center. Put each point on the
outer tyre edge where the rubber meets the ground. Do not use the rim, wheel
arch, tyre shadow, grass, road marking, or the bottom of the bounding box.
Preserve a real height difference between the two contact points when the
image has perspective or camera roll.

Treat the wheel-center line and tyre-contact line as two distinct measured
calibration lines. Apparent tyre radii can differ in a perspective image, so
the two lines can have different slopes. Do not force the contact points to
equal `wheel_center_y + one shared radius`. Python uses the wheel-center line
as the upper longitudinal constraint and the two tyre contacts as the lower
constraint; both must become horizontal after rectification. This deliberate
keystone correction is required even when it is stronger than a simple image
rotation.

Use actual door seams, pillars, and the visible A-pillar/windshield edge to
determine the lateral perspective direction. The windshield edge runs from
the front roof corner down to the windshield base/cowl. Do not substitute the
hood slope, roof outline, grille edge, or an invented diagonal for it.

`body_chassis_line_1000` is a semantic hint for the fixed vehicle body's real
lower structural baseline. Put its endpoints on the same physical rocker/sill
plane beneath the bed, cab, and engine section. Follow the visible door-sill
lower edge, rocker-panel lower edge, or a factory running board only when that
running board defines the continuous fixed lower body plane.

When a dark rocker strip, side step, or running board has both an upper and a
lower horizontal edge, select its lower edge. On a lifted pickup this edge is
usually below the wheel-center line but still clearly above the tyre contact
line. Do not choose a longer silver door crease or decorative strip above it.

This line is not the ground line and is not the lowest visible pixel. Do not
use tyre bottoms, wheel centers, wheel arches, suspension, axle, exhaust,
cast shadow, road edge, parking stripe, bumper bottom, or a decorative door
moulding. Lifted trucks can have much greater ground clearance than standard
trucks, so measure this line from actual mechanical edges instead of using a
fixed fraction of image height. If several horizontal edges exist, choose the
lowest long edge belonging continuously to the fixed bed/cab/hood body, not
an isolated accessory or shadow. Python uses this only as a search hint and
snaps it to a real image edge.

Before returning this line, inspect pixels immediately above and below it:
the upper side must belong to fixed body metal, rocker, sill, or its fixed
lower trim; the lower side must transition toward underbody/clearance. Reject
a candidate when body paint continues substantially below it, because that is
a styling crease. Reject a lower candidate that runs through tyres or their
shadow, because that is a wheel/ground tangent. Prefer several collinear
mechanical edge fragments over one unrelated but longer photographic line.
Both endpoints must remain inside the horizontal interval between the two
wheel centers, preferably about 10% inward from each wheel center. The line
is normally level with or below the wheel-center line. A candidate clearly
above both wheel centers is usually a door crease and must be rejected.

`background_type` describes the image outside the vehicle:

- `white`: a solid white or near-white studio background;
- `transparent`: the supplied image has a transparent background;
- `environment`: any real scene, including road, grass, trees, buildings,
  parking lines, shadows, or other photographic surroundings.

Classify a white vehicle photographed outdoors as `environment`; this field
describes the background, not the vehicle paint.
`front` is exactly `left`, `right`, or `unknown`.

Do not return any other fields.
