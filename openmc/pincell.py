import openmc
import numpy as np
import math

def define_pincell(pitch, region_outer_radii, region_materials, n_sectors_per_region, n_rings_per_region, name):

    universe = openmc.Universe(name=name)

    corner_radius = math.sqrt( (pitch/2.0)**2 + (pitch/2.0)**2 )

    n_regions = len(region_materials)

    # Working from inner -> outwards
    for r in range(n_regions):
        n_sub_regions = n_rings_per_region[r]

        r_inner = 0.0
        r_outer = corner_radius

        if r > 0 :
            r_inner = region_outer_radii[r-1]
        if r < n_regions - 1:
            r_outer = region_outer_radii[r]
    
        total_area = np.pi * (r_outer ** 2) - np.pi * (r_inner ** 2)
        sub_region_area = total_area / n_sub_regions

        radius_delta = (r_outer - r_inner) / n_sub_regions
        if n_sub_regions < 3 :
            radius_delta *= (2.0/3.0)

        r_last = r_inner

        inner_cylinder = openmc.ZCylinder(x0=0, y0=0, r=r_last)

        for sr in range(n_sub_regions):
            # Compute next radius outwards
            r_next = math.sqrt( r_last**2 + sub_region_area / np.pi )

            if r == n_regions - 1 :
                r_next = r_last + radius_delta
            
            # If the next one is the final one, we manually set it to avoid changing it due to floating point rounding
            if sr == n_sub_regions -1 :
                r_next = r_outer

            # Create outer cylinder
            outer_cylinder = openmc.ZCylinder(x0=0, y0=0, r=r_next)

            n_sectors = n_sectors_per_region[r]

            theta_last = np.pi / 4.0
            theta_delta = (np.pi * 2.0) / n_sectors
            a = math.cos(theta_last + np.pi/2.0)
            b = math.sin(theta_last + np.pi/2.0)
            plane_last = openmc.Plane(a=a, b=b)
            for sec in range(n_sectors):
                cell = openmc.Cell()
                cell.fill = region_materials[r]

                theta_next = theta_last + theta_delta

                # If the next one is the final one, we manually set it to avoid changing it due to floating point rounding
                if sec == n_sectors - 1 :
                    theta_next = np.pi / 4.0

                # define next plane surface
                a = math.cos(theta_next + np.pi/2.0)
                b = math.sin(theta_next + np.pi/2.0)
                plane_next = openmc.Plane(a=a, b=b)

                # Typical case bounded by both inner and outer cylinders
                if (r_last > 0.0) and (r_next != corner_radius):
                    cell.region = +inner_cylinder & -outer_cylinder & -plane_last & +plane_next

                # Case in which neither inner nor outer cylinders are used (e.g., only 1 region in problem)
                elif (r_last == 0) and (r_next == corner_radius):
                    cell.region = -plane_last & +plane_next

                # Case in which we are at the innermost region
                elif r_last == 0 :
                    cell.region = -outer_cylinder & -plane_last & +plane_next

                # Case in which we are at the outermost region
                elif r_next == corner_radius :
                    cell.region = +inner_cylinder & -plane_last & +plane_next

                theta_last = theta_next
                plane_last = plane_next

                universe.add_cell(cell)

            # Advance the last radius
            r_last = r_next
            inner_cylinder = outer_cylinder

    return universe
